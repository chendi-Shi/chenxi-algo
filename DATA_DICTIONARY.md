# 输入字段与会计口径

输入 CSV 使用 UTF-8 / UTF-8 BOM，保留股票代码为字符串。金额不得加千位分隔符；数值空白、NA、N/A、null 视为未知，不填零。所有资料使用 YYYY-MM-DD 日期，按截止日当天结束处理。`ticker` 去除首尾空白并转大写；推荐标准化为 `600000.SH`、`000001.SZ`、`00700.HK` 等包含交易所的标识。

## statements.csv

每行是一家公司、一份年度报表的一个版本。期间必须为 350–380 天，相邻年度首尾连续。三年报表的币种、市场与行业必须一致；行业重新分类或币种改变时应先提供可比历史。金额统一来自年度**合并**财报，不混用母公司单体报表。

| 字段 | 定义 / 约定 |
|---|---|
| ticker / name | 证券标识 / 公司名称；必须为非空文本 |
| market | `A` 或 `HK` |
| sector | 基金认可的统一行业分类；避免同义标签拆分同业组 |
| currency | 报表金额币种，如 CNY、HKD、USD；三年必须一致 |
| unit_scale | 原始金额乘数；元=1，千元=1000，百万元=1000000，万元=10000 |
| scope | 必须为 `consolidated` |
| period_start / period_end | 当年损益与现金流覆盖的年度起止日期；资产负债为期末余额 |
| available_at | 本版本真正披露且可用的日期；不要用期末日期替代 |
| revision_id | 同日版本的非负整数，默认 0；仅用于同日内版本排序 |
| filing_id / source_url | 财报编号及 HTTP(S) 来源指针；本地资料可用基金获授权的文档链接 |
| net_income | 合并净利润，用于 ROA、现金利润质量与 F-score；报表利润口径为本地适配，不完全等同原论文非常项目前净利润 |
| net_income_parent | 归属于普通股母公司股东的净利润；ROE 使用此值 |
| core_income_parent | 可比的归母经常性/扣非净利润；A 股扣非、港股 adjusted profit 必须由研究员核对调整项后统一，不直接混用定义 |
| operating_cash_flow | 合并经营活动现金流量净额，可为负 |
| total_assets | 合并总资产，已知时必须大于零 |
| equity_parent | 普通股归母权益；剔除少数股东、优先股及不适用资本工具，口径与净利润一致 |
| current_assets / current_liabilities | 流动资产 / 流动负债；非负。零流动负债的比率为未知 |
| long_term_debt | 长期有息债务总额，**包含重分类为一年内到期的长期债务**；不是仅“非流动负债合计” |
| total_debt | 总有息债务；租赁负债是否纳入应统一，并与现金流、利息口径一致 |
| cash | 可动用现金及现金等价物；受限现金应剔除或另外人工核验 |
| revenue / cogs | 营业收入 / 对应销售成本；收入已知时必须正，成本非负；不是净利润或期间费用 |
| capex | 现金资本开支正数，程序计算 `CFO - capex`；现金流表负号需先映射，不使用固定资产余额变动代替 |
| ebit / interest_expense | 同口径 EBIT / 利息费用正数；零利息覆盖率为未知。只有已核验 total_debt=0 且 interest_expense=0 才以明确无债标记参与排序；期末无银行贷款不等于全年无利息，也不等于没有租赁债务 |
| equity_issued | `1`=本年度发行普通股，`0`=核验无发行，空白=未知；应记录发行事件而非净股数差或发行减回购的净现金流 |
| audit_opinion | `unqualified`=已核验无保留意见；其他或空白转人工复核。持续经营强调事项、治理风险还需阅读原文 |

可缺少部分数字字段，程序保留未知及指标覆盖率；缺少关键数据不能成为候选。连续三年并不代表三年的比较口径已自动可信，合并范围、重述及会计政策变化需人工核对。应将新财报披露的重述历史作为新的版本，保留其新 `available_at`。

ROA=`当年合并净利润 / 上年末总资产`；CFO/资产使用相同分母；杠杆=`长期有息债务 / 两年平均总资产`；周转=`收入 / 上年末总资产`。原论文正文与表格的周转分母存在差异，本实现统一采用正文的期初资产，详见 RESEARCH.md。

## valuations.csv

| 字段 | 定义 / 约定 |
|---|---|
| ticker | 与财务数据对应的证券标识 |
| snapshot_date / available_at | 估值观察日 / 此版本可用日，后者不得早于前者 |
| revision_id | 同日版本号，默认 0 |
| market_cap / unit_scale / currency | **全公司全部普通股类别的总市值**、金额乘数、总市值币种；正值 |
| cap_scope | 必须为 `total_company`；A/H 总市值应将不同类别按各自价格及汇率合并后输入 |
| source_url | 总市值数据来源 HTTP(S) 指针 |
| fx_to_reporting | 若币种不同，1 单位总市值币种兑换为多少报表币种。例如 HKD → CNY 输入 CNY/HKD |
| fx_date / fx_source_url | 跨币种时必须提供不晚于截止日的近期 FX 日期及来源；同币种可空 |

转换：`market_cap_reporting = market_cap × unit_scale × fx_to_reporting`。默认估值和 FX 距截止日不得超过 7 天，期间财报不得早于截止日 550 天以上。超过一年提示复核；可通过 `expected_latest_period` 指定已确认应当可用的年度期末，缺少该年度则进入复核。估值比率使用年度数据；尚未实施最新中报/TTM。

F-score 默认诊断（`min_f_score: null`）；显式设置数字才启用完整九项门槛。CFO/利润与扣非/归母在质量排名中分别封顶 2 和 1，原值保留；这是未经效果校准的研究规则。ROE 要求期初和期末归母权益均正。总有息债务不得小于包含一年内到期部分的长期有息债务。关闭评分模块不关闭净债务基础门槛。

## documents.jsonl

每行一个 JSON 对象。必填非空文本字段为 document_id、ticker、available_at、source_url、text；source_type 取 annual_report、announcement、transcript、alternative_data。page 为原始资料页码，可空。没有来源的记录或未来文档进入拒绝日志。

```json
{"document_id":"report-001-p88","ticker":"00700.HK","available_at":"2026-03-20","source_url":"https://example.invalid/replace-with-authorized-source","source_type":"annual_report","page":88,"text":"替换为获授权原文片段；保留否定和上下文。"}
```

示例地址仅说明格式，不是真实来源。推荐每条为一个完整段落，便于定位；程序保留最多 1000 字符摘录及原始行。TF-IDF 英文分词、中文字符二元组只是词面检索，不能理解业务事实。精确重复按同 ticker 去除，余弦相似度 ≥ 0.90 提示人工近重复核对。关键词只给主题，不能把“没有关联交易”判为存在关联交易。

## 输出状态

| 状态 | 含义 |
|---|---|
| candidate | 完整基础数据、通过基本面门槛、同业综合分达到配置阈值；优先展开研究 |
| watchlist | 已通过基础门槛但优先级较低，或评分字段/同业样本不足 |
| data_review | 关键数据、时间、版本、审计意见或估值口径需要核验 |
| specialist_review | 行业不适合通用模型，需要银行/保险/金融/REIT 专用分析 |
| excluded | 完整的可检查数据未通过初筛门槛；可按投资风格调配置 |

综合分只在同市场同行业有意义；市场和行业之间不做全局排名。PCA 与文本提示不影响任何状态。JSON 中 null 表示未知，CSV 空单元格与零值区分，F-score 不完整时只提供已知项数与通过项下界。
