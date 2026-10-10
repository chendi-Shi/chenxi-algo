# 主题检索公开证据样本

本语料以 **2026-10-10** 为观察截止日，供“光模块、机器人、AI 服务器”检索流程的可复现小样本验收。包含 9 家选定的 A 股、港股科技与制造公司、9 条原文短引、9 份官方来源，其中 6 份有完整 PDF 导入清单。这是人工选取的验证样本，不是全市场股票池，也不代表历史时点完整上市公司范围。

## 文件和口径

- `data/theme_companies.json`：公司身份、研究行业标签、`scope` 和 `universe_as_of`。行业标签是本项目研究分类，不冒称交易所行业分类。
- `data/theme_documents.jsonl`：一行一条原文短引，包含来源、可用日、PDF 物理页或网页章节、原文件 SHA256；原文只做换行空白整理。
- `data/theme_sources.json`：来源元数据、抓取文件大小、缓存路径、原文短引 SHA256、作者转述及人工验证标签。`summary_basis=analyst_paraphrase` 明确区分转述与原文。
- `data/theme_ingest_manifest.json`：6 份完整 PDF 的下载和导入清单，`local_path` 相对于该清单所在的 `data/` 目录。PDF 本体和提取全文留在被忽略的本地目录，不随公共仓库发布。

## 来源

| 公司 | 来源 | 日期依据 | 原文定位 | 选入用途 |
|---|---|---|---|---|
| 中际旭创 300308.SZ | [Cloud & AI Infrastructure](https://www.zj-innolight.com/en/cloud_ai_infrastructure.html) | 本次观察 2026-10-10 | 同名网页章节 | 光收发模块产品证据 |
| 新易盛 300502.SZ | [2024 年报](https://static.cninfo.com.cn/finalpage/2025-04-23/1223219348.PDF) | 披露 2025-04-23 | PDF 物理第 11 页 | 光模块研发、生产及销售 |
| 光迅科技 002281.SZ | [Cloud and Enterprise](https://www.accelink.com/lighting_your_dreams/CloudandEnterprise.html) | 本次观察 2026-10-10 | 同名网页章节 | 光模块产品证据 |
| 埃斯顿 002747.SZ | [2024 年报](https://static.cninfo.com.cn/finalpage/2025-04-29/1223370517.PDF) | 披露 2025-04-29 | PDF 物理第 13 页 | 自有工业机器人产品 |
| 优必选 09880.HK | [官方产品首页](https://www.ubtrobot.com/cn) | 本次观察 2026-10-10 | 消费级智能硬件 | 机器人技术及产品 |
| 联想集团 00992.HK | [ThinkSystem 产品组合](https://lenovopress.lenovo.com/lp1553.pdf) | 动态 PDF，本次观察 2026-10-10 | PDF 物理第 10 页 | AI 服务器产品证据 |
| 浪潮信息 000977.SZ | [2024 年报](https://static.cninfo.com.cn/finalpage/2025-03-29/1222950880.PDF) | 披露 2025-03-29 | PDF 物理第 11 页 | AI 通用服务器产品证据 |
| 上纬新材 688585.SH | [许可协议暨关联交易公告](https://static.cninfo.com.cn/finalpage/2025-12-06/1224855156.PDF) | 披露 2025-12-06 | PDF 物理第 1 页 | 机器人开发阶段，不能误判为已量产 |
| 岱美股份 603730.SH | [设立全资子公司公告](https://static.cninfo.com.cn/finalpage/2025-11-18/1224808933.PDF) | 披露 2025-11-18 | PDF 物理第 4 页 | 机器人技术准备阶段，不能误判为成熟业务 |

巨潮文件用官方披露 URL 中的日期作为 `available_at`，不使用年报报告期末替代披露日。无独立可核实发布日期的网页与动态 PDF 标记 `date_basis=observed_at`；其首次可见时间只保守设为本次观察日，不能用于更早日期的历史回测。`universe_as_of=2026-10-10` 只表明本次手选公司清单的观察日。

联想完整产品组合 PDF 当前缓存约 57 MiB；导入器单份源文件上限为 64 MiB。PDF 物理页从 1 起计，不使用可能不同的印刷页码。下载内容变动导致哈希不匹配时，应重新人工审核版本、短引和日期，不能直接覆盖旧哈希。

优必选年报原 URL 在本次运行中返回 HTTP 403，因此没有把该未取得的年报列入成功核验来源；采用可取得的官方产品页。曾考察的新易盛 2020 产品目录、联想已撤回的 SR685a V3 产品指南没有进入最终语料或导入清单。这里的 2024 年报也仍是历史报告期业务证据，后续研究应补查最新报告与公告。

## 如何复验

```powershell
python validation/validate_theme_sources.py
python validation/validate_theme_sources.py --verify-sources
```

第一条仅检查离线 schema、公司映射、日期、URL、哈希字段和清单一致性，输出 `mode=offline_metadata_only`、`original_sources_reverified=false`，不会联网，也不假装复验原文。

第二条核验本地 `validation/raw/theme/` 原文件 SHA256，并在指定 PDF 物理页或去除 script/style 后的网页正文中检索原文短引。仅允许 Unicode 兼容规范化和空白差异，不做模糊文本匹配；缓存缺失、哈希变化、页码错误或原文不存在都会失败。PDF 核验需要 `pypdf`。本次已完成 **9/9 份缓存原文哈希及定位核验**。它验证的是缓存版本，不证明线上内容没有更新。

完整正文导入使用仓库的 `ingest_theme.py`，并将输出写到被忽略的 `output/`：

```powershell
python ingest_theme.py --manifest data/theme_ingest_manifest.json --output output/theme_full_documents.jsonl --download
```

清单中已有全部官方 URL 与已核验的源文件哈希。命令优先读取清单 `local_path` 指向的原文件，缺失时按清单下载并检查 SHA256；下载缓存的实际路径见生成的导入 manifest。详细选项见 `python ingest_theme.py --help`。网页当前只提供精选短引，不计入完整 PDF 覆盖范围。

来源验证器的四项核心离线回归覆盖“不复验时不能声称原文通过”、篡改短引、未来可用日、篡改原文件；另外两项 CLI 回归检查保存实际核验结果和解析失败时替换旧报告。全部使用标准库及临时微型 HTML 原文件，不需要网络或大 PDF：

```powershell
python -m unittest discover -s tests -p test_theme_sources.py -v
```

## 评估标签的适用范围

`evaluation_cases` 为三条查询的人工标签，每条包含 `query`、`direct_tickers`、`development_tickers`、`non_direct_tickers`。标签在运行检索前编写，覆盖这 9 条精选证据：

- `direct_tickers`：该原文直接说明相应产品或业务。
- `development_tickers`：主题相关，但该公告时点处于开发或技术准备阶段；是“成熟业务”的反例，并非“主题相关性”的反例。
- `non_direct_tickers`：所选片段没有直接产品证据，不等于已证明整家公司完全无关。全文检索可能揭示更多产业链联系，因此全文评估必须另行核对证据，不能机械复用这些片段标签。

上纬、岱美的两份历史公告不证明后续一直没有进展；检索展示必须保留日期并提示查阅后续公告。官方产品页及企业宣传用语是发行人陈述，不能自动升级为第三方核实的市场地位、订单、客户关系或未来盈利结论。本语料也没有为这些公司填充财务数据，因此单凭这些证据不能完成成长、估值、经营改善或亏损分类。

9 条人工精选证据的命中率仅是流程检验；不得报告为全市场检索精确率、召回率、投资有效性或企业生产验收通过率。扩大股票池、加入独立标注的完整文档与未见样本、检查更新覆盖后，才能评估实际研究效率。
