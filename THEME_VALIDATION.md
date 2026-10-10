# v0.2 主题发现流程验证（2026-10-10，历史记录）

本记录对应 v0.2 研究原型与当时的代码哈希，测试数量不代表当前版本。后续生产化工作与新验证记录见 [PRODUCTION.md](PRODUCTION.md)。尚未完成全市场覆盖、真实科技制造财务数据库准入或投资研究有效性验收。

## 已完成的验证

| 检查 | 范围和结果 |
|---|---|
| 软件回归与独立财务复算 | 最终231项测试通过（223项主测试+8项A股来源测试，无跳过）；结果见 [theme_enterprise_results.json](validation/theme_enterprise_results.json)；另以独立公式核对1,800项指标和200个综合分 |
| 原文件证据复验 | 9家公司、9个公开来源；文件SHA256及指定页短引全部通过，见 [theme_source_results.json](validation/theme_source_results.json)；两份开发阶段公告另做页面渲染目视检查 |
| 完整PDF导入 | 6份原文件，共669页，零空文本页，哈希全部匹配；本机约61秒；另通过一次显式HTTPS下载、SHA256核对及8页导入的联网测试 |
| 精选段落主题召回 | 3主题×9公司共27个主题/公司组合，9个已标相关组合均召回，18个未标相关组合未召回 |
| 精选段落业务阶段 | 7个已标开展业务组合中5个被识别为直接业务，2个保守保留为不确定；2个开发阶段案例未误标直接业务 |
| 全文检索 | 669页PDF加3条官网短引补充；三个主题的已知相关组合均找回，两个开发阶段案例未误标直接业务；完整结果见 [theme_retrieval_results.json](validation/theme_retrieval_results.json) |
| 查询与数据稳定性 | 6个中英文/简繁/全角别名在精选与全文中保持公司集合和关系一致；改变输入排序、加入未来资料或重复证据均不改变结果 |
| 财务端到端 | 完全虚构的8家公司样例覆盖成长、相对价值、改善、亏损和未分类五类；真实公开证据集没有配套财务历史，明确显示未分类 |
| 导出一致性 | 检查输入和代码变化、文件哈希、完整标记、CSV与JSON逐字段一致；中途失败不能标记为完整；不完整的PDF导入结果被主流程拒绝 |

这里的精选段落成绩来自开发时使用的小集合，不能解释为全市场100%准确率。全文新增了浪潮信息的机器人相关提及；没有完整人工真值，故列为待审，不据精选短句将其算作误报。两个短证据业务阶段漏判是保守规则的实际限制，未隐藏或改写真值。

## 实测发现并修复的问题

- 对手、客户、行业背景、工商经营范围的提及可能被误作公司主营业务；现在分别降为需核验的线索。
- 已停止或不再经营的业务不能凭旧式经营措辞独立召回；“尚未量产但正在开发”仍保留观察。
- PDF在中文词内换行可能绕过阶段识别；分类时处理这些断行，证据保持原文及原位置。
- 同公告跨页出现“业务开展”“组建团队”，可能掩盖另一页明确的整体开发阶段说明；现在限定同来源同日期同主题做核对，附阶段证据。新型号研发不会覆盖已量产产品线。
- 证券代码大小写、全角查询、来源观察日期可能在模块间丢失；现在统一代码、支持查询规范化，保留date_basis。
- 极端金额相减可能溢出并误产生改善信号；现在非有限差值为未知。
- CSV财务标签若错误而文件哈希正确，原先仍可能验证通过；现在逐字段与JSON独立核对。

## 复现

```powershell
python -m pip install numpy==2.2.6 pypdf==6.10.0
python validation/enterprise_checks.py --companies 1000 --output validation/output/enterprise_recheck.json
python validation/validate_theme_sources.py --output validation/output/source_recheck.json
python ingest_theme.py --manifest data/theme_ingest_manifest.json --output output/theme_full_documents.jsonl --download
python validation/validate_theme_retrieval.py --full-documents output/theme_full_documents.jsonl --output validation/output/retrieval_recheck.json
python discover.py --query "机器人" --companies data/theme_companies.json --documents output/theme_full_documents.jsonl --as-of 2026-10-10 --output output/full-robots
python discover.py --verify output/full-robots
```

来源验证默认只检查离线元数据；加 `--verify-sources` 才会复验原文件，要求按来源元数据中的缓存路径提供9份原文件。下载器使用 `output/source-cache` 内容哈希缓存；网页原文件不在PDF manifest中。公开网页变化时无法保证重新取得相同版本，历史缓存和观察日期应由团队维护。

CI在Windows/Linux、Python3.10/3.12上运行软件和离线公开样例检查；CI不联网抓取最新公告，也不重复下载完整PDF。团队使用前需要接入实际公司库、经核对的财务数据和独立人工标注集，并建立更新与复核流程。
