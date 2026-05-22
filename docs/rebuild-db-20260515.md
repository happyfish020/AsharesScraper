
No tasks in progress


我的生产db坏了，要全部重建，怎样重建D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2\docs\V8_TABLES_DAILY_AND_BACKFILL_RUNBOOK.md 的tables data ? 
Sunday 2:26 PM

我先看一下这份 runbook 和仓库里现有的重建脚本/命令，把可执行的重建步骤整理成一套适合生产库全量重建的流程。

Ran 1 command
你要直接去DB里比对， 在这部分是重建及前置条件的数据 又怎重建，它依赖  cn_local_industry_map_hist for 85xxxx.SI
Sunday 2:28 PM


Worked for 8m 55s
我已经直接查了你现在的生产库 cn_market_red，结论很明确：

cn_ts_sw_industry_master 正常：SW2014=28，SW2021=31
cn_ts_sw_industry_member_hist 也有数据：SW2014=5582，SW2021=5847
但 cn_local_industry_map_hist 只有 5195 行，而且 industry_id like '85%.SI' = 0 行
cn_v7_v8_industry_crosswalk、cn_v7_v8_industry_crosswalk_latest、cn_stock_v8_to_v7_sw_map_latest 目前都是空的
上游并不是没源，cn_board_member_map_d 里 85xxxx.SI 是有的，而且我直接查到覆盖从 2000-01-04 到 2026-05-15，2026-05-15 当天还有 7101 行 V8 成分映射
所以这次不是先修 runbook 里的 5 张表，而是要先把 cn_local_industry_map_hist 的 85xxxx.SI 重建出来。
关键点是：这一步要用 build_local_industry_map_hist.py，不要只跑 build_tushare_sw_replacement_sources.py 或 build_sw_industry_member_hist.py，因为后者只会补官方 801xxx.SI，补不出你要的 85xxxx.SI。

推荐重建顺序

如果你要整条链全量重建，按这个顺序：

python scripts\build_local_industry_map_hist.py ^
  --start 2000-01-04 ^
  --end 2026-05-15 ^
  --level L1 ^
  --src SW2021 ^
  --workers 4
python scripts\build_local_industry_proxy_daily.py ^
  --start 2000-01-04 ^
  --end 2026-05-15 ^
  --resume ^
  --workers 4
python scripts\build_tushare_sw_replacement_sources.py ^
  --start 2000-01-04 ^
  --end 2026-05-15 ^
  --srcs SW2021 SW2014 ^
  --levels L1 ^
  --replace-master ^
  --replace-members ^
  --output-dir reports\analysis\tushare_sw_replacement_sources_full_rebuild
python scripts\build_v7_v8_industry_crosswalk.py ^
  --start 2000-01-04 ^
  --end 2026-05-15 ^
  --replace ^
  --srcs SW2021 SW2014 ^
  --source-mode db ^
  --output-dir reports\analysis\sw_v7_v8_crosswalk_full_rebuild
python scripts\build_v7_v8_crosswalk_latest.py ^
  --replace ^
  --output-dir reports\analysis\v7_v8_crosswalk_latest
python scripts\build_stock_v8_to_v7_sw_map_latest.py ^
  --replace ^
  --output-dir reports\analysis\stock_v8_to_v7_sw_map_latest
先验收这 3 条 SQL

SELECT COUNT(*) AS cnt, COUNT(DISTINCT industry_id) AS industry_cnt
FROM cn_local_industry_map_hist
WHERE industry_id LIKE '85%.SI';
SELECT COUNT(*) AS cnt, MIN(effective_date) AS min_d, MAX(effective_date) AS max_d
FROM cn_v7_v8_industry_crosswalk;
SELECT src, COUNT(*) AS cnt, MAX(asof_date) AS asof_d
FROM cn_v7_v8_industry_crosswalk_latest
GROUP BY src;
补一句风险提示

我还查到 cn_board_industry_master 里目前 没有 85xxxx.SI 主数据，所以第一步虽然能把 85xxxx.SI membership 重建出来，但部分 industry_name 可能先回落成代码值。如果你这次要求“先恢复可跑、可对齐”，上面这套足够；如果你还要求 85xxxx.SI 名称也完整，我下一步可以继续帮你把 85 系列名字来源补齐。

如果你愿意，我可以下一步直接替你在库里执行这整套重建，并一边跑一边把每一步的实际行数回报给你。



Sunday 2:37 PM
另一个问题是这些表的数据的下游历史也要重建， 怎建？
Sunday 2:43 PM


Worked for 1m 15s
要，而且要分两类重建：

历史表要全量重算
*_latest 快照表只要最后再刷一次
你这个场景里，先前那批源表重建完之后，真正要补的下游历史链主要是这些：

cn_local_industry_proxy_daily
cn_v7_v8_industry_crosswalk
cn_stock_fundamental_daily
cn_stock_quality_score_daily
cn_industry_capital_flow_daily
cn_ga_stock_role_map_daily
cn_stock_mainline_strength_daily
cn_ga_mainline_radar_daily
cn_ga_market_pulse_daily
cn_mainline_lifecycle_daily
cn_unified_alpha_score_daily
这些不是历史表，只需要最后重刷一次：

cn_v7_v8_industry_crosswalk_latest
cn_stock_v8_to_v7_sw_map_latest
cn_stock_leader_sw_l1_latest_snap
推荐重建顺序

先重建你前面提到的源链：

cn_local_industry_map_hist
cn_local_industry_proxy_daily
cn_ts_sw_industry_master
cn_ts_sw_industry_member_hist
cn_v7_v8_industry_crosswalk
然后再跑下游历史派生。最稳的做法是直接用统一入口，只开 derived：

set V8_BACKFILL_START=20100101
set V8_BACKFILL_END=20260515
set V8_BACKFILL_INCLUDE_PRICE=0
set V8_BACKFILL_INCLUDE_FUNDAMENTAL=0
set V8_BACKFILL_INCLUDE_DAILY_BASIC=0
set V8_BACKFILL_INCLUDE_SW_DAILY=0
set V8_BACKFILL_INCLUDE_BOARD_REFRESH=0
set V8_BACKFILL_INCLUDE_SW_HISTORY=0
set V8_BACKFILL_INCLUDE_CONCEPT_HISTORY=0
set V8_BACKFILL_INCLUDE_DERIVED=1
set V8_BACKFILL_INCLUDE_VALIDATIONS=1
set V8_BACKFILL_INCLUDE_CROSSWALK_LATEST=0
python runner.py --flag tu --tasks v8_backfill --asof 20260515
这一步会按项目里定义的派生顺序重建历史：

build_stock_fundamental_daily.py
build_stock_quality_score_daily.py
build_industry_capital_flow_daily.py
build_ga_stock_role_map_daily.py
build_cn_stock_mainline_strength_daily.py
build_ga_mainline_radar_daily.py
build_ga_market_pulse_daily.py
build_mainline_lifecycle_daily.py
再跑一轮 strength/radar/pulse
build_unified_alpha_score_daily.py
最后再刷快照表

先把 crosswalk 历史表重建好：

python scripts\build_v7_v8_industry_crosswalk.py ^
  --start 2000-01-04 ^
  --end 2026-05-15 ^
  --replace ^
  --srcs SW2021 SW2014 ^
  --source-mode db ^
  --output-dir reports\analysis\sw_v7_v8_crosswalk_full_rebuild
再刷 latest：

python scripts\build_v7_v8_crosswalk_latest.py --replace
python scripts\build_stock_v8_to_v7_sw_map_latest.py --replace
一句关键提醒

v8_backfill --include-derived=1 会重建下游历史派生，但不会替你补出 85xxxx.SI 的 cn_local_industry_map_hist。
所以顺序一定是：

先补 cn_local_industry_map_hist
再补 cn_local_industry_proxy_daily
再补 cn_v7_v8_industry_crosswalk
最后跑 v8_backfill 的 derived 历史链
最后刷 *_latest
如果你要，我下一条可以直接给你一份“生产库全量重建 checklist”，按执行顺序列成 1 到 10 步，带每一步的验收 SQL。



Sunday 2:44 PM
D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2\docs\V8_TASK_SPLIT_RUNBOOK_20260515.md 根据这个文档， 全面整理新文档，V7数据 或全空DB上 进行全部历史数据（2010-2016）的数据顺序 或命令， 除了 v7_v8_industry_crosswalk 外可能还有别的，需要补上 
Sunday 2:50 PM


Worked for 5m 27s
你不要到库里看实际情况，增量回补
我已经按你的要求整理成一份不依赖实际库状态、面向 2010-2016 增量回补的新文档了：

docs/V8_HISTORICAL_BACKFILL_2010_2016_INCREMENTAL_RUNBOOK.md

这份新 runbook 重点补上了旧文档里容易漏掉的几块：

v8_backfill 不会自动补齐的 V8 兼容映射链：
cn_local_industry_master
cn_local_industry_map_hist
cn_local_industry_proxy_daily
cn_ts_sw_industry_master
cn_ts_sw_industry_member_hist
cn_v7_v8_industry_crosswalk
以及最后的 *_latest 快照
空库预置需要的 DDL / GA schema / board membership SP
V7 库或空库从 2010-01-01 到 2016-12-31 的推荐顺序
明确采用增量回补思路：
优先 --refresh
优先 --resume
不默认做 destructive rebuild
大窗口按年切
我这次没有再去看库里的实际情况，文档是按仓库现有脚本和 runbook 定义整理的。

如果你要，我下一步可以继续帮你做两件事里的一个：

把旧的 V8_TASK_SPLIT_RUNBOOK_20260515.md 头部加一个“请改看新历史回补文档”的跳转说明。
再给你补一版“只保留最短执行命令清单”的 operator 版。

V8_HISTORICAL_BACKFILL_2010_2016_RUNBOOK.md
Document · MD
Open

V8_HISTORICAL_BACKFILL_2010_2016_INCREMENTAL_RUNBOOK.md
Document · MD
Open

V8_TASK_SPLIT_RUNBOOK_20260515.md
Document · MD
Open

Edited 2 files
+983
-0
Undo
Review
docs/V8_HISTORICAL_BACKFILL_2010_2016_RUNBOOK.md
docs/V8_HISTORICAL_BACKFILL_2010_2016_INCREMENTAL_RUNBOOK.md


Sunday 2:56 PM
在当前的数据的环境中，需要补什么？ 我已经回补了80%, 
Sunday 3:01 PM

不看库里的实际状态、只按当前仓库定义来判断的话，你“已经回补了 80%”之后，最可能还差的不是原始行情，而是兼容映射链和下游派生链。

优先补这几组。

V8 兼容映射链
cn_local_industry_master
cn_local_industry_map_hist
cn_local_industry_proxy_daily
cn_ts_sw_industry_master
cn_ts_sw_industry_member_hist
cn_v7_v8_industry_crosswalk
这一组最容易被漏掉。v8_backfill 本身不会把这条链完整补齐。

下游历史派生链
cn_stock_fundamental_daily
cn_stock_quality_score_daily
cn_industry_capital_flow_daily
cn_ga_stock_role_map_daily
cn_stock_mainline_strength_daily
cn_ga_mainline_radar_daily
cn_ga_market_pulse_daily
cn_mainline_lifecycle_daily
cn_unified_alpha_score_daily
如果你原始数据已经补了大半，这一组通常就是剩下那 20% 的主体。

latest 快照层
cn_v7_v8_industry_crosswalk_latest
cn_stock_v8_to_v7_sw_map_latest
cn_stock_leader_sw_l1_latest_snap
这组不用全历史，只要在历史链跑完后最后刷一次。

空库/半空库常见漏项
sp_refresh_board_member_hist
sp_build_board_member_map
leader score 相关 views
GA schema / lifecycle schema
如果这些没齐，后面很多脚本会“能跑但补不全”。

如果你问“现在最该先补哪几个”，我的建议顺序是：

cn_local_industry_map_hist
cn_local_industry_proxy_daily
cn_ts_sw_industry_*
cn_v7_v8_industry_crosswalk
全部 derived history
最后刷 latest snapshots
如果你想做一个不查库、只按工程顺序的最小补尾清单，就跑这几段：

python scripts/build_local_industry_map_hist.py --start 2010-01-01 --end 2016-12-31 --level L1 --src SW2021 --resume --workers 4
python scripts/build_local_industry_proxy_daily.py --start 2010-01-01 --end 2016-12-31 --resume --workers 4
python scripts/build_tushare_sw_replacement_sources.py --start 2010-01-01 --end 2016-12-31 --srcs SW2021 SW2014 --levels L1 --replace-master --replace-members --output-dir reports/analysis/tushare_sw_replacement_sources_2010_2016
python scripts/build_v7_v8_industry_crosswalk.py --start 2010-01-01 --end 2016-12-31 --replace --srcs SW2021 SW2014 --source-mode db --output-dir reports/analysis/sw_v7_v8_crosswalk_2010_2016
set V8_BACKFILL_START=20100101
set V8_BACKFILL_END=20161231
set V8_BACKFILL_INCLUDE_PRICE=0
set V8_BACKFILL_INCLUDE_FUNDAMENTAL=0
set V8_BACKFILL_INCLUDE_DAILY_BASIC=0
set V8_BACKFILL_INCLUDE_SW_DAILY=0
set V8_BACKFILL_INCLUDE_BOARD_REFRESH=0
set V8_BACKFILL_INCLUDE_DERIVED=1
set V8_BACKFILL_INCLUDE_CROSSWALK_LATEST=0
python runner.py --flag tu --tasks v8_backfill --asof 20161231
python scripts/build_v7_v8_crosswalk_latest.py --replace
python scripts/build_stock_v8_to_v7_sw_map_latest.py --replace
如果你愿意，我下一条可以直接把“你已经补了 80% 之后，只剩最后 20%”压缩成一个 最后补尾 checklist。



Sunday 3:01 PM
