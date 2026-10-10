# 每日基本面研究流程：本机试运行与生产准入

当前状态：**生产化改造中，尚未通过企业生产验收**。用户已确定每天更新 A/H 公司资料，按产品或主题输出有来源的研究名单；先在本机跑通，部署位置后定。仓库公开的是程序、合成测试和有限公开验证记录；公司内部资料与下载的完整原文留在本地忽略目录。

新增的 `semantic_discover.py` 是显式启用的语义检索实验入口，真实使用 E5 与 mMARCO 模型；尚未接入每日任务或取得研究团队验收。它沿用数据、主题业务时效和财务规则，输出始终声明 `production_approved: false`。模型对照实验、已发现的误判及使用方式见 [ALGORITHM_EXPERIMENTS.md](ALGORITHM_EXPERIMENTS.md)。

## 已确定的研究行为

- 范围为 A 股及港股的科技、制造公司。证券名录、业务分类、财务可比行业分别保留依据；交易所的宽行业代码不能直接当作细分同业组。
- 输入光模块、机器人、AI 服务器等产品或主题，寻找公司自身业务及产业链线索。BM25 决定证据阅读顺序，不能理解为收入占比或投资价值。
- 盈利公司可进入优质成长、相对价值、经营改善多份名单。三个盈利风格都要求有当前业务支持；规划、主体不明或新否定与旧支持冲突的公司暂存待核验线索。
- 亏损但业务相关的公司保留在单独观察名单。未知财务字段不补零，数据缺失不等于公司质量差。
- 输出原文、URL、页码、披露日期及日期依据、文件哈希、业务状态和逐条财务门槛。

## 程序分层

| 环节 | 入口 | 约束 |
|---|---|---|
| 官方证券名录 | `universe_sources.py` | 原始响应快照、分页完整性、日期与证券类型核验、离线重放 |
| 公告链接发现 | `disclosure_sources.py` | 按证券及时间窗查询，保留原始响应、披露时间和每家公司失败原因 |
| 原文导入 | `ingest_theme.py` | HTTPS、哈希缓存、有限重试、PDF 页码、并发锁及完整提交标记 |
| 单主题研究 | `discover.py` | 原文检索、业务阶段与否定证据、财务风格、数据覆盖门槛 |
| 多主题批次 | `research_job.py` | 所有主题读取同一份输入快照，保留每次运行和完整导出 |
| 一次每日更新 | `daily_update.py` | 名录核验→公告发现→原文导入→研究批次；失败不能冒充当日更新成功 |

每个每日任务只执行一次；可以由未来确定的公司调度器每日启动。当前没有安装 Windows 计划任务、服务器服务或 Codex 定时任务。

## 运行环境与检查

Python 3.10+。核心检索、财务计算与调度只使用标准库；PDF 提取需要 pypdf。CI 固定验证版本，安装示例：

```powershell
python -m pip install pypdf==6.10.0
python -m unittest discover -s tests -v
```

`numpy==2.2.6` 是原有可选 ML 检查和完整 CI 使用的依赖。单次导入：

```powershell
python ingest_theme.py --manifest data/theme_ingest_manifest.json --output output/public/corpus.jsonl --download
```

缓存默认固定原始版本，SHA256 不符即拒绝。`--refresh --download` 明确重取 URL；配置了本地文件时仍以该文件为来源。缓存存在不代表今天确认过上游内容。没有文本的扫描 PDF 会报出缺口，当前没有 OCR。

导入生成 `corpus.jsonl`、`corpus.jsonl.manifest.json`、`corpus.jsonl.complete.json`。批处理要求三件套完整且 manifest 状态为 `completed`。部分成功的导入也不作为完整语料发布。

## 本机每日更新试跑

仓库的 [daily_local.example.json](config/daily_local.example.json) 只选择两家深市光通信公司及年度报告，用于检查真实来源链路，**不是团队已批准的研究池或生产政策**。配套政策要求该小池 100% 的披露、经营财务和估值覆盖。样例没有填入虚构财务，因此即便公告和检索正常，财务门槛仍应阻断正式发布。

```powershell
python daily_update.py --config config/daily_local.example.json --download
python daily_update.py --status output/daily_local_pilot
```

`--download` 才允许联网。任务按 Asia/Shanghai 的运行当日计算截止日；`lookback_days=365` 表示从截止日减 365 天起、两端包含，共 366 个日历日。可以加 `--as-of 2026-10-10` 固定试验截止日，但新的名录不能用于更早的历史日期。配置也支持固定 `start_date/end_date`，它们与 `lookback_days` 互斥。

退出码 0 表示本次任务成功且数据门槛通过，3 表示数据或范围阻断，2 表示执行失败。详细原因见该次 `job.json`、阶段报告和研究输出；`--status` 的退出码只表示状态文件能否核验，不表示最新任务成功。

需要离线重放时，在新配置中指明上次运行内的 `universe_snapshot`、`disclosure_snapshot` 和 `document_cache` 三个目录，不加 `--download`。只有公告查询响应而没有 PDF 缓存时，不能完成原文导入。新输出目录不能覆盖这些输入快照。

当前程序每次重取指定时间窗，尚未实现保留全部历史的增量资料库。滚动窗口移出的资料不会自动进入新任务，不能将该模式当作完整历史研究数据库。首次覆盖范围与后续增量规则需在生产化下一阶段落实。

本机 2026-10-10 真实试跑取得中际旭创、新易盛两份年报，共 399 页（229、170 页）。“光模块”和“800G 光模块”均找到两家公司；经营财务与估值覆盖不足使任务停在 `blocked`，没有更新成功指针。此次联网运行约 154 秒，包括下载、PDF 提取和两条主题查询。结果仍保留需人工审阅的正反陈述冲突。

完整记录见 [daily_live_check.json](validation/daily_live_check.json)。随后在禁止网络请求和 PDF 下载的条件下离线重放，语料文件哈希一致，两条查询结果在排除运行标识后完全一致；财务与估值缺失仍然阻断发布。原始失败、修复后联网运行、A/H 日期阻断与离线重放四次审计均保留，并通过文件完整性核验。

第一次实际试跑暴露 Windows 长路径写入失败；修复使用程序内部的扩展绝对路径，没有修改系统设置，随后在相同目录深度重试。超过 260 字符路径及输入/输出目录嵌套冲突已加入实际文件系统回归。真实 UNC 共享盘尚未测试。

## 公司库、财务数据和准入策略

正式公司库必须明确 `ticker/name/market/sector/scope/universe_as_of`。`scope` 为 technology 或 manufacturing；`sector` 是经核对的细分财务同业组。`export_scope_candidates` 输出官方宽行业对应的候选队列，故意保留未经核验的 `scope/sector=null`；它不能直接替代可用研究池。

财务 CSV 按 [DATA_DICTIONARY.md](DATA_DICTIONARY.md) 对接。引擎要求连续三份可比年度合并报表及来源、可用日期和审计信息。港股的扣非口径、有息债务、利息分类以及 A/H 全发行人总市值不能仅凭字段名自动认定相同。当前每日公告流程**尚未自动把全 A/H 财报转换成已经核对的财务表，也没有实时估值供应商**。缺少这些输入时，研究输出会显示未分类，正式批次被数据门槛阻断。

`production_policy` 是一个独立 JSON 文件，所有键必须显式填写，无内置“生产通过”阈值：

| 字段 | 含义 |
|---|---|
| `expected_tickers` | 实际声明的研究证券池，非本次搜索命中的公司 |
| `max_universe_age_days` | 名录最大允许天数 |
| `max_document_age_days` | 至少一份合格披露的最大允许天数 |
| `min_document_coverage` | 声明池内合格披露覆盖比例，0–1 且大于 0 |
| `min_operating_coverage` | 经营财务完整覆盖比例 |
| `min_valuation_coverage` | 估值财务完整覆盖比例 |
| `max_invalid_records` | 允许的错误记录数量 |
| `max_identity_conflicts` | 允许的证券/文档身份冲突数量 |
| `require_document_hashes` | 是否要求原文件 SHA256 |
| `require_known_date_basis` | 是否要求 official_release 或 observed_at 日期依据 |

`observed_at` 只说明何时观察到网页；不能证明业务从该日开始，不能据此推翻有披露日期的官方否定公告。文档覆盖只衡量指定条件下“有可用资料”，不保证已经取得每个应披露期间的完整年报、中报和临时公告。后者仍需独立清单对账。

`theme_business_freshness` 另查当前查询对应的经营支持原文，使用同一 `max_document_age_days` 以及哈希、日期依据要求。新的董事会公告不能让三年前的机器人业务证据变新。未满足主题证据时效的盈利风格被暂缓，原始财务判定与历史线索保留；亏损观察不因此丢失。最新支持原文在 `business_support_evidence` 中单独保留，避免被相关度排序或展示数量上限截掉。

数据检查通过记为 `data_gates_passed`。所有输出中的 `production_approved` 仍为 false：通过文件和数据门槛，不等于企业已确认研究质量、运行容量或部署验收。研究门槛见 [THEME_ALGORITHM.md](THEME_ALGORITHM.md)，默认值还未与投资团队的判断校准。

## 同一快照运行多个主题

在本地创建 `job.json`，所有路径相对于该配置文件：

```json
{
  "schema_version": 1,
  "as_of": "2026-10-10",
  "companies": "companies.json",
  "documents": "public/corpus.jsonl",
  "statements": "statements.csv",
  "valuations": "valuations.csv",
  "production_policy": "policy.json",
  "queries": [
    {"name": "optics", "query": "光模块"},
    {"name": "robots", "query": "机器人"},
    {"name": "ai_servers", "query": "AI服务器", "market": "HK"}
  ],
  "limit": 20,
  "output_root": "research_history"
}
```

`statements/valuations/theme_config/financial_config/limit` 可选；估值输入要求同时提供报表。少报表或估值仍可生成待补资料结果，但无法满足严格覆盖准入。日期必须明确填写。

```powershell
python research_job.py --config job.json
python research_job.py --status research_history
```

每次运行建独立 `runs/<run_id>`，复制输入、导入审计三件套和使用的源代码，保存阶段事件与耗时。一个主题失败后仍记录其余主题，整体返回失败；任何主题未通过数据门槛，整体被阻断。只有所有主题成功、导出完整且门槛通过，才原子更新 `last_success.json`；程序不删除历史。

读取最近结果应先运行 `--status`，检查最后一次尝试是否失败、成功指针指向哪一天，不能把历史成功结果展示成当天刷新结果。文件哈希能检测改变，不能防止有写权限的人连同审计记录一起重写；更强的防篡改存储应由部署环境提供。

## 真实来源覆盖与已知缺口

2026-10-10 的官方名录采集见 [universe_source_check.json](validation/universe_source_check.json)：

- SSE 2,320 条股类证券中包含 1 个 CDR；SZSE 2,904 条。普通股宽行业候选 3,970 条，其中制造 3,554、科技 416。候选数量不代表已取得这些公司的正文或财务数据。
- HKEX 返回 2,771 条选定港币股类证券，但下载表的生效日期是 2026-10-12，晚于此次截止日，所以被阻断。不能把未来生效表倒填为当前名录。
- 北交所尚未支持，实测官方 HTTPS 证书验证失败，没有关闭证书校验绕过。沪市的一次请求也出现连接拒绝，保留了 partial 记录；单次成功不证明每日稳定性。
- 当前公开全文检索验证仅覆盖 6 份 PDF 的 669 页及 3 条官网短引。开发标签中主题已知相关组合能召回，但已开展业务启发式只识别 7 个中的 5 个，其余保守待核验；这不是独立市场级准确率估计。

公告来源实测见 [disclosure_source_check.json](validation/disclosure_source_check.json)：3 家 A 股与 2 家港股的年报查询找到 5 份完整年报链接，原始响应可离线重放。较宽公告查询检查了 A 股多页与港股日期分窗；遇到声明为 PDF、链接实际指向展示网页的记录时报告失败。年报与 ESG 复合标签已纳入识别。年报模式刻意不包含所有临时公告，不能据此宣布重大事项无遗漏。

新的全文验证见 [production_retrieval_results.json](validation/production_retrieval_results.json)。单主题本机测量见 [production_query_benchmark.json](validation/production_query_benchmark.json)，只针对该小语料，不包含下载与 PDF 抽取。旧版 [THEME_VALIDATION.md](THEME_VALIDATION.md) 及其报告保留为历史记录。

2026-10-10 冻结源码后的本机软件检查见 [production_hardening_results.json](validation/production_hardening_results.json)：375 项测试通过，无失败、错误或跳过；独立算术核对包含 1,800 项财务指标及 200 个综合分数。20 组随机同业样本的宽束搜索与穷举结果一致，窄束搜索仅 16 组一致，不能保证最优。报告保存测试与源文件哈希，并确认验证期间代码未变化。这些检查验证程序行为，不验证投资有效性、全市场检索质量或生产容量。

## 生产验收仍需达到的条件

1. 明确目标证券池与经核对的细分行业；A/H 覆盖分别统计，不以几家公司样例替代全市场。
2. 核对各市场公告查询的分页、历史可得性及原始披露清单；包含最新年报、中报、重大业务变化。证明失败重跑不会漏公告或静默保留旧版本。
3. 为目标池接入并抽样逐项对账三年财务、审计意见与全公司市值，保留修订版本和实际可用日期；完成港股会计映射。
4. 由研究人员在独立标注集检查主题相关性、发行人主体、业务阶段、否定证据和亏损观察；记录误报、漏报并确定可接受范围。
5. 用真实目标数据量测试单次更新耗时、检索延迟、内存和磁盘增长；单元测试和合成规模测试不能替代这项。
6. 在确认的部署环境连续试运行，演练源中断、进程崩溃、磁盘写入失败、恢复和历史还原，指定维护与数据复核责任。

当前没有为以上未完成项作通过声明，也未安装或启动后台生产服务。
