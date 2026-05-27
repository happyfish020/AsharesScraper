# V8 Task Split Runbook — Daily / Weekly / Monthly

## Goal

Split AshareScraperV2 V8 refresh jobs into independent sub-tasks. Each sub-task can be rerun alone, and the bat files now execute sub-tasks sequentially instead of hiding everything behind one large task.

## Date window rule

`runner.py` now supports explicit date ranges:

```bat
python runner.py --flag tu --tasks v8_daily_market_raw --start-date 2026-05-01 --end-date 2026-05-15
python runner.py --flag tu --tasks v8_rotation_repair --start_date 20260501 --end_date 20260515
```

Aliases are supported:

- `--start-date` / `--start_date`
- `--end-date` / `--end_date`

If no explicit range is supplied, the old automatic mode is preserved:

```bat
python runner.py --flag tu --tasks v8_daily --asof latest --days 1
```

That resolves to `[latest - days + 1, latest]`.

## Daily split

`daily.bat` now runs these independent tasks in order:

1. `v8_daily_market_raw` — stock/index daily raw refresh
2. `v8_daily_reference` — SW industry daily, rotation sector snapshot, daily events
3. `v8_rotation_repair` — rotation audit + upstream/downstream auto-repair
4. `v8_daily_audit` — stock/index coverage audit + targeted index repair
5. `v8_daily_derived_foundation` — stock fundamental daily, quality score, capital flow, role map
6. `v8_daily_derived_mainline` — stock/mainline strength, radar, pulse, lifecycle + validations
7. `v8_daily_derived_alpha` — unified alpha + validations + latest crosswalk

Default:

```bat
daily.bat
```

Manual range:

```bat
daily.bat --start-date 2026-05-01 --end-date 2026-05-15

or 
python runner.py --flag tu --tasks v8_daily_market_raw --start-date ...
python runner.py --flag tu --tasks v8_daily_reference --start-date ...
python runner.py --flag tu --tasks v8_rotation_repair --start-date ...
...

```

Break-fix examples:

```bat
python runner.py --flag tu --tasks v8_daily_market_raw --start-date 2026-05-01 --end-date 2026-05-15
python runner.py --flag tu --tasks v8_rotation_repair --start-date 2026-05-01 --end-date 2026-05-15
python runner.py --flag tu --tasks v8_daily_derived_alpha --start-date 2026-05-01 --end-date 2026-05-15
整体流水线
v8_daily_market_raw
        ↓
v8_daily_reference
        ↓
v8_rotation_repair
        ↓
v8_daily_audit
        ↓
v8_daily_derived_foundation
        ↓
v8_daily_derived_mainline
        ↓
v8_daily_derived_alpha
1. v8_daily_market_raw

原始市场数据层（Raw Market Data）

作用：更新最基础、最原始的行情与事件数据。

典型内容：

cn_stock_daily_price
cn_index_daily_price
cn_stock_daily_basic
cn_sw_industry_daily
cn_event_disclosure_date
cn_event_earnings_forecast

特点：

数据直接来自 Tushare / Eastmoney 等。
不做复杂计算。
所有后续任务的基础。
2. v8_daily_reference

参考映射层（Reference & Mapping Data）

作用：更新相对低频变化的静态或半静态数据。

典型内容：

cn_stock_basic
cn_stock_name_change
cn_local_industry_map_hist
cn_v7_v8_industry_crosswalk
cn_board_member_map_d

特点：

主要是股票、行业、板块的映射关系。
通常变化不大。
3. v8_rotation_repair

板块轮动与快照层（Rotation Snapshot）

作用：计算行业/概念轮动排名和最新快照。

典型内容：

cn_sector_rot_bt_daily_t
SP_REPAIR_ROT_BT_NAV
SP_REFRESH_ROTATION_SNAP_ALL
rotation_snapshot

输出：

strongest sectors
top concepts
rotation signals
4. v8_daily_audit

数据质量审计层（Data Audit）

作用：检查上游数据是否完整。

典型内容：

缺失日期检查
行数阈值检查
空表检查
freshness 检查

输出：

cn_ga_data_readiness_daily
5. v8_daily_derived_foundation

基础衍生层（Foundation Derived Data）

作用：从原始数据构建中间指标。

典型内容：

cn_stock_monthly_basic
cn_stock_fundamental_quality_snap
cn_market_structure_daily

特点：

为 Mainline 和 Alpha 提供通用因子。
6. v8_daily_derived_mainline

主线识别层（Mainline Engine）

作用：识别当前市场主线和龙头角色。

典型内容：

cn_ga_mainline_radar_daily
cn_ga_stock_role_map_daily
cn_mainline_strength_daily

输出：

主线名称
生命周期阶段
龙头/跟风角色
7. v8_daily_derived_alpha

统一 Alpha 层（Unified Alpha）

作用：生成最终选股评分。

典型内容：

cn_stock_leader_score_daily
cn_unified_alpha_score_daily

输出：

综合评分
排名
Alpha bucket
与 GrowthAlpha V8 的对应关系
数据层	对应 V8 系统
Market Raw	基础行情
Reference	行业/板块映射
Rotation Repair	板块轮动
Audit	数据质量控制
Derived Foundation	通用因子
Derived Mainline	主线识别
Derived Alpha	统一选股评分
运行方式

日常更新：

daily.bat

历史回补：

daily.bat --start-date 2010-01-01 --end-date 2026-03-30
最终产出

执行完后，V8 会得到：

最新市场原始数据；
最新行业与板块映射；
最新板块轮动快照；
数据质量报告；
基础因子；
当前市场主线；
最终 Unified Alpha 排名。
一句话理解
Task Group	简单理解
v8_daily_market_raw	下载原始行情和事件
v8_daily_reference	更新行业/板块映射
v8_rotation_repair	计算板块轮动快照
v8_daily_audit	检查数据是否完整
v8_daily_derived_foundation	生成基础因子
v8_daily_derived_mainline	识别当前主线
v8_daily_derived_alpha	生成最终选股评分

```

## Weekly split

`weekly.bat` now runs:

1. `v8_weekly_refresh` — board membership, stock basic, periodic events
2. `v8_weekly_audit` — coverage audit + targeted index repair
3. `v8_weekly_finalize` — latest leader snapshot + crosswalk latest

Default:

```bat
weekly.bat
```

Manual range:

```bat
weekly.bat --start-date 2026-04-01 --end-date 2026-05-15
Weekly Pipeline 总体结构
v8_weekly_refresh
        ↓
v8_weekly_audit
        ↓
v8_weekly_finalize
1. v8_weekly_refresh
低频基础数据刷新

负责更新变化频率较低、但对 V8 很重要的数据。

建议内容
Event 类（低频）
cn_event_dividend
cn_event_shareholder_change
cn_event_unlock
cn_event_buyback
Board Membership
cn_board_concept_member_hist
cn_board_industry_member_hist
cn_board_member_map_d
Reference Mapping
cn_stock_basic
cn_stock_name_change
cn_local_industry_map_hist
cn_v7_v8_industry_crosswalk
Optional Historical Repairs
对最近 30–90 天进行补缺。
2. v8_weekly_audit
周级数据质量审计
审核内容
空表检查
最新日期检查
行数异常
重复记录检查
数据覆盖率
输出表
cn_ga_data_readiness_daily
weekly_audit_report.csv
3. v8_weekly_finalize
周级后处理与优化
建议内容
数据压缩/归档
清理 staging tables
归档旧临时表
Statistics Refresh
ANALYZE TABLE
OPTIMIZE TABLE（可选）
Snapshot Refresh
更新 latest views
更新缓存表
Collation Check（可选）
检查新增表是否使用正确 collation。
推荐 weekly.bat
@echo off
for %%T in (
    v8_weekly_refresh
    v8_weekly_audit
    v8_weekly_finalize
) do (
    echo [WEEKLY] %%T %*
    python runner.py --flag tu --tasks %%T %*
    if errorlevel 1 goto :fail
)
goto :eof

:fail
echo Weekly pipeline failed.
exit /b 1
各阶段职责概览
Task	作用
v8_weekly_refresh	更新低频 reference/event/board 数据
v8_weekly_audit	审核完整性与质量
v8_weekly_finalize	清理、统计刷新、归档
与 Daily 的区别
Daily	Weekly
高频原始数据	低频 reference/event
Rotation/Mainline/Alpha	Board/Event/Mapping
每日必跑	每周一次即可
推荐放到 Weekly 的内容
Board membership
Concept/industry member history
Dividend
Shareholder changes
Crosswalk mapping
数据维护与归档
最终建议

是的，以下结构非常适合：

v8_weekly_refresh
v8_weekly_audit
v8_weekly_finalize

其中：

refresh = 更新低频数据；
audit = 数据质量检查；
finalize = 清理、归档、统计优化。

```

## Monthly split

`monthly.bat` now runs:

1. `v8_monthly_refresh` — monthly/financial refresh + periodic events
2. `v8_monthly_audit` — coverage audit + targeted index repair
3. `v8_monthly_derived` — downstream derived rebuild + validations + crosswalk

Default:

```bat
monthly.bat
Monthly Pipeline 总体结构
v8_monthly_refresh
        ↓
v8_monthly_audit
        ↓
v8_monthly_derived
1. v8_monthly_refresh
月度基础财务数据刷新

主要负责更新低频财务和估值数据。

财务报表
cn_stock_income
cn_stock_balancesheet
cn_stock_cashflow
cn_stock_fina_indicator
月度估值
cn_stock_monthly_basic
财务质量参数
cn_fundamental_quality_param_t
2. v8_monthly_audit
财务数据完整性审计
检查内容
最近报告期覆盖率
股票覆盖率
空值比例
异常值检查
行数阈值
输出
cn_ga_data_readiness_daily
monthly_financial_audit.csv
3. v8_monthly_derived
财务衍生因子构建
典型输出
cn_stock_fundamental_quality_snap
cn_stock_quality_scores
cn_unified_alpha_score_daily（可选重新刷新）
主要因子
ROE
ROIC
毛利率
净利增长
现金流质量
杠杆指标
Monthly Pipeline 流程
财务报表下载
      ↓
完整性审计
      ↓
质量评分
      ↓
基本面快照
      ↓
Unified Alpha 更新（可选）
monthly.bat 示例
for %%T in (
    v8_monthly_refresh
    v8_monthly_audit
    v8_monthly_derived
) do (
    echo [MONTHLY] %%T %*
    python runner.py --flag tu --tasks %%T %*
    if errorlevel 1 goto :fail
)
与 Daily / Weekly 的区别
Pipeline	重点
Daily	行情、轮动、主线、Alpha
Weekly	Event、Board、Reference
Monthly	财务报表与质量因子
Monthly 主要产出
最新财务数据；
基本面质量评分；
月度 quality snapshot；
Alpha 基本面因子。
推荐执行时间
每月 1–5 日；
或财报季（4/30、8/31、10/31）后重点运行。
最终建议

你提出的结构是最佳实践：

v8_monthly_refresh
v8_monthly_audit
v8_monthly_derived
一句话总结
Task	含义
v8_monthly_refresh	更新财务报表与月度估值
v8_monthly_audit	检查财务数据完整性
v8_monthly_derived	构建财务质量与基本面因子

```

Manual range:

```bat
monthly.bat --start-date 2026-01-01 --end-date 2026-05-15
```

## Frequency placement

Daily:

- stock daily price
- index daily price
- SW industry daily
- rotation sector snapshot / BT repair
- daily events
- derived V8 daily layers

Weekly:

- board/industry/concept membership refresh
- stock basic/reference refresh
- periodic events
- latest leader snapshot/crosswalk refresh

Monthly:

- financial statements
- stock monthly basic
- fundamental monthly rebuild
- monthly downstream derived rebuild when enabled

## Safety rule

The old combo tasks remain available:

```bat
python runner.py --flag tu --tasks v8_daily --asof latest --days 1
python runner.py --flag tu --tasks v8_weekly --asof latest --days 7
python runner.py --flag tu --tasks v8_monthly --asof latest --days 31
```

Use the split bat files for production operation and the combo tasks for quick compatibility checks.


历史回补不要先跑 daily。

推荐顺序：

1. monthly.bat
2. weekly.bat
3. daily.bat

原因：

行为说明
--refresh
强制重新拉取指定范围数据
不依赖“已存在则跳过”的增量逻辑
--replace
在写入前删除指定日期范围已有数据
然后重新插入


正确顺序应该是：

1. daily_by_year.bat / daily.bat --start-date ... --refresh
2. index 历史回补，如需要
3. weekly_by_year.bat 审计与生成 weekly

monthly = 财务/质量/基本面底座
weekly  = 板块成员/事件/映射底座
daily   = 行情/轮动/主线/Alpha 最终日级产物

推荐命令：

monthly.bat --start-date 2010-01-01 --end-date 2026-03-30
   monthly_split.bat  --start-date 2010-01-01 --end-date 2026-05-15
weekly.bat  --start-date 2010-01-01 --end-date 2026-03-30
daily.bat   --start-date 2010-01-01 --end-date 2026-03-30

更稳一点：

monthly.bat --start-date 2010-01-01 --end-date 2026-03-30
weekly.bat  --start-date 2010-01-01 --end-date 2026-03-30
daily.bat   --start-date 2010-01-01 --end-date 2026-03-30
daily.bat

审核 ：
python runner.py --flag tu --tasks v8_full_coverage_audit --start-date 2010-01-01 --end-date 2026-05-15
daily.bat --start-date 2010-01-01 --end-date 2010-12-31 --refresh

输出报告位置：

reports/v8_full_coverage_audit/

包含：

空表检查
日期覆盖范围检查
日级连续性缺口
月级连续性缺口
重复 key 风险
缺失日期明细 CSV

 

已覆盖。这个 v8_full_coverage_audit 的核心就是查 中间跳隔缺数据，不是只看 MIN/MAX。

它会做：

实际表内日期
vs
交易日历/自然月日历

然后输出中间缺口明细。

示例：

cn_stock_daily_price
2010-01-04 有
2010-01-05 缺
2010-01-06 有

会被识别为 gap。

运行：
weekly 审核 ！
python runner.py --flag tu --tasks v8_full_coverage_audit --start-date 2010-01-01 --end-date 2026-05-15

重点看报告里的：

reports/v8_full_coverage_audit/*missing_dates*.csv
reports/v8_full_coverage_audit/*gap*.csv

结论：中间断层/跳隔缺数据是这个审核工具的重点检查项。

所以补数顺序应该是：
weekly_by_year.bat --start-date 2010-01-01     --end-date 2010-12-31 --refresh --replace

daily.bat 先跑到 rotation_snapshot
然后单独重建 / 修复 rotation snapshot
再重跑 daily.
for price only 
stock_daily_price_by_year.bat 2010 2012
python runner.py --flag tu --tasks index --start-date 2010-01-01 --end-date 2026-05-15 --refresh 

核心不是补原始行情，而是补 snapshot 派生表。

你现在先找我修复后生成的明细 CSV：

reports/.../rotation_repair_snap_missing_detail_*.csv

看里面缺的是哪些字段，一般会是：

trade_date
industry_id / board_id
snapshot_type
missing_reason
source_table

然后按明细补。

独立财务回补工具包，用于更新以下 4 张表：

cn_stock_fina_indicator
cn_stock_income
cn_stock_balancesheet
cn_stock_fundamental_daily

下载：

下载 stock_fundamental_backfill_tool.zip

支持参数
stock_fundamental_backfill.bat --start-date 2007-01-01 --end-date 2012-12-31 --replace

或：

stock_fundamental_backfill.bat --start-date 2007-01-01 --end-date 2026-05-15 --replace
执行顺序
stock_fundamental.income
stock_fundamental.balancesheet
stock_fundamental.fina_indicator
scripts/build_stock_fundamental_daily.py --replace
推荐命令（解决你当前问题）
stock_fundamental_backfill.bat --start-date 2007-01-01 --end-date 2012-12-31 --replace

完成后，再运行：

daily_by_year.bat --start-date 2011-01-01 --end-date 2026-05-15 --refresh --replace

独立 event_backfill.bat，支持 --start-date / --end-date / --replace，默认走你现有 runner.py --tasks 入口，不改 daily/weekly/monthly。

Thought for a couple of seconds

已生成：

下载 event_backfill_tool.zip

用法：

event_backfill.bat --start-date 2010-01-01 --end-date 2026-05-15 --replace <=====

说明：--replace 会被 bat 接收，但不会传给 runner.py，因为当前 runner.py 不支持全局 --replace。

从你贴出的 SP 看，有一批明显是旧 V7 / sector rotation 实验链路，可以不作为 V8 核心依赖。真正和 V8 当前主线/板块基础数据有关的，主要只保留 2～3 个。

必须保留 / V8 仍有价值
SP	判断
sp_build_board_member_map	保留。生成 cn_board_member_map_d，V8 主线、行业、概念映射核心表。
sp_refresh_board_member_hist	保留。把 staging 成员表转成历史成员表，供 sp_build_board_member_map 使用。
sp_refresh_stock_universe_status	可保留。不是板块核心，但对 active/delisted 股票过滤有用。

依据：sp_build_board_member_map 明确从 cn_board_concept_member_hist / cn_board_industry_member_hist 生成 cn_board_member_map_d；sp_refresh_board_member_hist 明确维护这两张 hist 表。

可以不用 / 更像旧 V7 sector-rotation 链路

这些基本围绕 cn_sector_rotation_*、cn_sector_rot_*、cn_rotation_*_snap_t，看起来是旧的行业轮动实验/快照/回测链路，不是 V8 当前 Unified Alpha / Mainline 基础层必须项：

SP	建议
SP_BACKFILL_ROT_BT_FROM_PRICE	可不用，旧 rotation 回测 NAV 补全
SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE	可不用，旧 rotation ranked 表
SP_BUILD_SECTOR_ROTATION_RANKED_LATEST	可不用
SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE	可不用，旧 rotation signal
SP_BUILD_SECTOR_ROTATION_SIGNAL_LATEST	可不用
SP_REFRESH_ROTATION_SNAP_ALL	可不用，旧 rotation entry/holding/exit snapshot
SP_REPAIR_ROT_BT_NAV	可不用
SP_ROTATION_DAILY_REFRESH	可不用，旧 rotation 一键刷新入口
SP_VALIDATE_AGAINST_BASELINE	可不用，旧 sector rotation baseline 验证
SP_VALIDATE_SECTOR_ROT_RUN	可不用
sp_rebuild_rotation_20200101_20260228	可不用，硬编码日期，明显是一次性旧任务
暂时保留但降级为 research/legacy
SP	判断
sp_refresh_sector_eod_hist	暂保留。它基于 cn_board_member_map_d + price 生成 cn_sector_eod_hist_t，可作为 V8 行业/概念强度研究输入，但不要让它成为 A-line 必需依赖。
sp_backfill_sector_eod_hist_monthly	暂保留。只是 sp_refresh_sector_eod_hist 的月度批处理封装。
最简 V8 必需链路
CALL sp_refresh_board_member_hist(...);
CALL sp_build_board_member_map('2010-01-01', '2026-03-30');

其余 SP_*ROTATION*、SP_*SECTOR_ROT*、SP_*BT*、SP_VALIDATE_SECTOR_ROT_RUN 基本都可以标记为：

legacy_v7_rotation / not_required_for_v8_core

先不要删，建议改名归档或写入 docs/LEGACY_SP_INVENTORY.md，避免 V8 daily runner 误调用。


大部分 cn_event_* 表对 GrowthAlpha V8 当前核心（A-line + B-line）并不是必需依赖。

一、我的总体建议
✅ 当前阶段建议：保留基础事件表，但停止重度构建
必须保留（轻量）
cn_event_calendar
cn_event_earnings
cn_event_dividend
可暂停更新（大多数）
其他复杂 cn_event_* 表
二、V8 当前真正依赖程度
功能模块	是否依赖事件表
E77c / E122 / E146_v4 A-line	❌ 不依赖
Unified Alpha Score	❌ 不依赖
Mainline Detection	❌ 不依赖
Industry Alpha Overlay	❌ 不依赖
Lifecycle State	❌ 不依赖
B-line Long Cycle Bull Discovery	❌ 不依赖
财报前风险过滤（未来可选）	⚠️ 可选
分红处理（未来可选）	⚠️ 可选
三、目前最常见但非核心的事件表
基础事件
cn_event_calendar
cn_event_earnings
cn_event_dividend
扩展事件
cn_event_buyback
cn_event_lockup
cn_event_insider_trade
cn_event_mna
cn_event_guidance
cn_event_macro
四、哪些值得保留
1. cn_event_earnings

用于：

财报日前后波动风险控制
避免业绩雷
2. cn_event_dividend

用于：

长期收益修正
回测准确性
3. cn_event_calendar

统一事件日历入口。

五、哪些可以停用

以下多数暂时不影响 V8 核心表现：

cn_event_buyback
cn_event_lockup
cn_event_insider_trade
cn_event_macro
cn_event_guidance
cn_event_mna
六、性能影响

EventLoaderDaily 往往会：

检查多个 cn_event_* 表
自动回补历史
逐日更新

这会明显拖慢 daily/weekly/yearly 任务。

七、当前最优策略
开发阶段（推荐）

只保留：

cn_event_calendar
cn_event_earnings
cn_event_dividend

其它全部停更。

更激进方案

如果当前专注于：

主线识别
行业轮动
Unified Alpha
生命周期

则可以暂时完全关闭 EventLoaderDaily。

八、未来什么时候再启用

当你开始做以下功能时再启用：

财报前减仓
除权除息精确处理
大股东减持风险
宏观事件管理
九、推荐优先级
Tier 1（保留）
cn_event_earnings
cn_event_dividend
cn_event_calendar
Tier 2（暂停）
其余所有 cn_event_*
十、我的最终建议

当前 V8 开发阶段，可以暂停绝大多数 cn_event_* 表的构建。


cn_mainline_strength_daily --- V8 不使用


##############################################################################33
final :
空库重建，建议完整顺序如下：

0. Schema / DDL
1. 基础主数据
2. 日线价格/指数/财务/事件
3. 行业基础层
4. 衍生基础层
5. 主线/生命周期/Unified Alpha
6. Audit
7. weekly / monthly
空库重建推荐顺序
0. 先建表 / 迁移 Schema
python scripts/apply_ga_p0_schema_migration.py --db-host 127.0.0.1 --db-port 3306 --db-user cn_opr_red --db-password sec_Bobo123 --db-name cn_market_red

python scripts/build_board_member_map_full.py --start 2010-01-01 --end 2026-03-31 --resume --db-host 127.0.0.1 --db-port 3306 --db-user cn_opr_red --db-password sec_Bobo123 --db-name cn_market_red


1. 先跑 daily_by_year，全量基础数据
daily_by_year.bat 2010 2026

但这里建议先让 daily_by_year 只负责：

stock_basic / stock_daily_price / stock_daily_basic
index_daily
fundamental
event

不要先跑 derived mainline validation。

2. 行业基础层，全量一次性跑

等 daily 基础数据至少价格和股票基础表完成后，跑：
 
 
python scripts/backfill_sw_industry_daily.py --start 2010-01-01 --end 2025-12-31 --resume

python scripts/build_local_industry_map_hist.py --start 2010-01-01 --end 2025-12-31 --level L1 --resume
python scripts/build_local_industry_map_hist.py --start 2010-01-01 --end 2025-12-31 --level L1 --resume
UPDATE cn_local_industry_map_hist
SET in_date = COALESCE(valid_from, '2010-01-01')
WHERE in_date IS NULL;

UPDATE cn_local_industry_map_hist
SET out_date = COALESCE(valid_to, '9999-12-31')
WHERE out_date IS NULL;

python scripts/build_local_industry_proxy_daily.py --start 2010-01-01 --end 2025-12-31 --resume --workers 4
 
python scripts/audit_cn_market_data_assets.py --db-host 127.0.0.1 --db-port 3306 --db-user cn_opr_red --db-password sec_Bobo123 --as-of-date 2026-03-31 --write-db 
 
 
3. 再跑 daily_by_year 的 derived 部分

如果你的 daily_by_year.bat 当前是一条龙，那建议拆成两轮：
-


推荐先跑：

python scripts/build_board_member_map_full.py --start 2010-01-01 --end 2026-03-31 --resume --db-host 127.0.0.1 --db-port 3306 --db-user cn_opr_red --db-password sec_Bobo123 --db-name cn_market_red

如要强制重建：

python scripts/build_board_member_map_full.py --start 2010-01-01 --end 2026-03-31 --replace --db-host 127.0.0.1 --db-port 3306 --db-user cn_opr_red --db-password sec_Bobo123 --db-name cn_market_red

第一轮：

只跑基础数据 / reference / event / fundamental

第二轮：

derived_foundation
derived_mainline
quality_score
industry_capital_flow
mainline_strength
mainline_lifecycle
unified_alpha
validation

如果目前没拆，就在完成第 2 步后重新跑：

daily_by_year.bat 2010 2026

这次让它补齐 derived 表。

4. 全局审计
python scripts/audit_cn_market_data_assets.py --db-host 127.0.0.1 --db-port 3306 --db-user cn_opr_red --db-password sec_Bobo123 --db-name cn_market_red --as-of-date 2026-03-31 --output-dir reports/data_audit --write-db --fail-on-critical
5. weekly_by_year
weekly_by_year.bat 2010 2026
6. monthly_by_year
monthly_by_year.bat 2010 2026
最终答案

你现在空库重建时，不是先跑行业三件套，因为它们依赖股票日线/基础表。

正确是：

apply schema
→ daily_by_year 基础层
→ 行业三件套
→ daily_by_year derived 层 --replace --refresh
→ audit
→ weekly_by_year
→ monthly_by_year

如果你的 daily_by_year 不能分基础层/derived 层，那就：

apply schema
→ daily_by_year 跑一遍，允许 derived 部分部分失败/跳过
→ 行业三件套
→ daily_by_year 再跑一遍补 derived
→ audit   -- 
    python runner.py --flag tu --tasks v8_full_coverage_audit --start-date 2010-01-01 --end-date 2026-03-30
	
	
	
第一轮（Raw Only）示例
set V8_DAILY_TASKS=v8_daily_market_raw,v8_daily_audit
set V8_DAILY_TASKS=v8_daily_market_raw,v8_daily_audit
 
set V8_DAILY_TASKS=v8_daily_derived_foundation
daily.bat 2010 --refresh
D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2>daily_init_phase2_by_year.bat 2019 2026 --refresh
《===================	


daily.bat 2014 --refresh
第二轮 2A（Foundation）

set V8_DAILY_TASKS=v8_daily_derived_foundation
daily.bat 2010 --refresh
daily.bat 2010 --refresh
daily_init_phase2_by_year.bat 2010 2010 --refresh


daily.bat 2014 --refresh
第二轮 2B（Mainline + Alpha）
set V8_DAILY_TASKS=v8_daily_derived_mainline,v8_daily_derived_alpha
daily.bat 2014 --refresh	
	
	
	set V8_DAILY_TASKS=v8_daily_derived_mainline,v8_daily_derived_alpha
daily_init_phase2_by_year.bat 2016 2018 --refresh
	
	set V8_DAILY_TASKS=v8_daily_derived_mainline,v8_daily_derived_alpha
	daily_init_phase2_by_year.bat 2010 2015 --refresh《==============?
	>daily_init_phase2_by_year.bat 2016 2018 --refresh 《================ing
	 set V8_DAILY_TASKS=v8_daily_derived_mainline,v8_daily_derived_alpha
daily_init_phase2_by_year.bat 2010 2015 --refresh
2a+2b
set V8_DAILY_TASKS=v8_daily_derived_foundation,v8_daily_derived_mainline,v8_daily_derived_alpha
daily_init_phase2_by_year.bat 2019 2026 --refresh <==================


set V8_DAILY_TASKS=v8_daily_derived_foundation,v8_daily_derived_mainline,v8_daily_derived_alpha
daily_init_phase2_by_year.bat 2023 2023 --refresh
set V8_DAILY_TASKS=v8_daily_derived_foundation,v8_daily_derived_mainline,v8_daily_derived_alpha
daily_init_phase2_by_year.bat 2024 2024 --refresh <==================
set V8_DAILY_TASKS=v8_daily_derived_foundation,v8_daily_derived_mainline,v8_daily_derived_alpha
daily_init_phase2_by_year.bat 2025 2025 --refresh
set V8_DAILY_TASKS=v8_daily_derived_foundation,v8_daily_derived_mainline,v8_daily_derived_alpha
daily_init_phase2_by_year.bat 2026 2026 --refresh
	
→ weekly_by_year 2022 2026 --replace --refresh
→ monthly_by_year  2022 2026  --replace --refresh


weekly_by_year 2014 2016 2022 done
monthly.bat --start-date 2010-01-01 --end-date 2016-12-31 --refresh running 
monthly.bat --start-date 2017-01-01 --end-date 2022-12-31 --refresh running 

weekly_by_year 2017 2019 running
weekly_by_year 2022 2022 running done
weekly_by_year 2022 2022 running done

weekly.bat --start-date 2020-01-01 --end-date 2021-12-31 --replace --refresh done
			
python scripts/build_industry_capital_flow_daily.py ^
  --start 2014-01-01 ^
  --end 2014-12-31 ^
  --replace
  
  ？？
===
审核 cn_board_member_map_d！！！！
python scripts/build_board_member_map_full.py ^
  --start 2010-01-01 ^
  --end 2026-03-31 ^
  --resume ^
  --dry-run ^
  --db-host 127.0.0.1 ^
  --db-port 3306 ^
  --db-user cn_opr_red ^
  --db-password sec_Bobo123 ^
  --db-name cn_market_red
  
  
第一轮最快建议
set V8_SKIP_ROTATION_SNAPSHOT=1
set V8_SKIP_EVENT_LOADER=1
set V8_SKIP_STOCK_QUALITY_SCORE=1
set V8_SKIP_INDUSTRY_CAPITAL_FLOW=1
set V8_SKIP_GA_STOCK_ROLE_MAP=1
set V8_SKIP_MAINLINE_STRENGTH=1
set V8_SKIP_MAINLINE_RADAR=1
set V8_SKIP_MARKET_PULSE=1
set V8_SKIP_MAINLINE_LIFECYCLE=1
set V8_SKIP_UNIFIED_ALPHA=1
set V8_DAILY_INCLUDE_CROSSWALK_LATEST=0
set V8_DAILY_INCLUDE_VALIDATIONS=0

REM 2A：foundation
set V8_SKIP_ROTATION_SNAPSHOT=1
set V8_SKIP_EVENT_LOADER=1
set V8_SKIP_STOCK_FUNDAMENTAL_DAILY=1

set V8_SKIP_STOCK_QUALITY_SCORE=0
set V8_SKIP_INDUSTRY_CAPITAL_FLOW=0

set V8_SKIP_GA_STOCK_ROLE_MAP=1
set V8_SKIP_MAINLINE_STRENGTH=1
set V8_SKIP_MAINLINE_RADAR=1
set V8_SKIP_MARKET_PULSE=1
set V8_SKIP_MAINLINE_LIFECYCLE=1
set V8_SKIP_UNIFIED_ALPHA=1

set V8_DAILY_INCLUDE_VALIDATIONS=0
set V8_DAILY_INCLUDE_CROSSWALK_LATEST=0


daily_by_year.bat 2010 2026 --refresh

REM 2B：mainline + alpha
set V8_SKIP_ROTATION_SNAPSHOT=1
set V8_SKIP_EVENT_LOADER=1
set V8_SKIP_STOCK_FUNDAMENTAL_DAILY=1

set V8_SKIP_STOCK_QUALITY_SCORE=1
set V8_SKIP_INDUSTRY_CAPITAL_FLOW=1

set V8_SKIP_GA_STOCK_ROLE_MAP=0
set V8_SKIP_MAINLINE_STRENGTH=0
set V8_SKIP_MAINLINE_RADAR=0
set V8_SKIP_MARKET_PULSE=0
set V8_SKIP_MAINLINE_LIFECYCLE=0
set V8_SKIP_UNIFIED_ALPHA=0

set V8_DAILY_INCLUDE_VALIDATIONS=0
set V8_DAILY_INCLUDE_CROSSWALK_LATEST=0
daily_by_year.bat 2010 2026 --refresh  


===============
包含：

set V8_STOCK_DAILY_MODE=spot_repair
set V8_DAILY_REPAIR_LOOKBACK_DAYS=15
set V8_DAILY_TASKS=v8_daily_market_raw,v8_daily_reference,v8_daily_audit,v8_daily_derived_foundation,v8_daily_derived_mainline,v8_daily_derived_alpha

daily_spot.bat
daily_spot_by_year.bat
daily_spot_update.bat
日常更新用法
daily_spot_update.bat

功能：

使用 spot_repair 模式
自动检查并修复最近 15 个交易日缺口
按日期批量更新（非逐股票扫描）
自动执行完整 V8 pipeline



那第二轮后不要直接 v8_full_coverage_audit。

如果 Monthly / Weekly 还没跑，顺序应改成：

python runner.py --flag tu --tasks v8_monthly_market_raw,v8_monthly_reference,v8_monthly_audit,v8_monthly_derived_foundation,v8_monthly_derived_mainline,v8_monthly_derived_alpha --start-date 2010-01-01 --end-date 2026-03-30 --replace --refresh

然后跑 weekly：

python runner.py --flag tu --tasks v8_weekly_market_raw,v8_weekly_reference,v8_weekly_audit,v8_weekly_derived_foundation,v8_weekly_derived_mainline,v8_weekly_derived_alpha --start-date 2010-01-01 --end-date 2026-03-30 --replace --refresh

最后再做总审计：

python runner.py --flag tu --tasks v8_full_coverage_audit --start-date 2010-01-01 --end-date 2026-03-30

一句话：先补 Monthly + Weekly，再跑 full coverage audit。



单独补 cn_mainline_lifecycle_daily：

python scripts/build_mainline_lifecycle_daily.py --start 2026-04-16 --end 2026-05-16 --replace --chunk-months 1 --db-host localhost --db-port 3306 --db-user cn_opr_red --db-name cn_market_red --db-password sec_Bobo123

但它依赖先有：

cn_ga_mainline_radar_daily

所以推荐顺序是：

python scripts/build_ga_mainline_radar_daily.py --start 2026-04-16 --end 2026-05-16 --replace --chunk-months 1 --db-host localhost --db-port 3306 --db-user cn_opr_red --db-name cn_market_red --db-password sec_Bobo123

python scripts/build_ga_market_pulse_daily.py --start 2026-04-16 --end 2026-05-16 --replace --chunk-months 3 --db-host localhost --db-port 3306 --db-user cn_opr_red --db-name cn_market_red --db-password sec_Bobo123

python scripts/build_mainline_lifecycle_daily.py    2026-04-16 --end 2026-05-16s  --replace   --chunk-months 1   --min-member-count 1   --db-host localhost   --db-port 3306   --db-user cn_opr_red   --db-name cn_market_red   --db-password sec_Bobo123

补完验证：

python scripts/validate_mainline_lifecycle_daily.py --start 2026-04-16 --end 2026-05-16 --min-rows 1 --db-host localhost --db-port 3306 --db-user cn_opr_red --db-name cn_market_red --db-password sec_Bobo123

先查缺失范围：

SELECT MIN(trade_date), MAX(trade_date), COUNT(*)
FROM cn_mainline_lifecycle_daily
WHERE trade_date BETWEEN '2026-04-16' AND '2026-05-16';


- cn_stock_mainline_strength_daily.trade_date: RANGE_NOT_COVERED | available=2026-04-17~2026-04-30, required=2026-04-17~2026-05-15
- cn_ga_mainline_radar_daily.trade_date: RANGE_NOT_COVERED | available=2026-04-17~2026-04-30, required=2026-04-17~2026-05-15
- cn_ga_stock_role_map_daily.trade_date: RANGE_NOT_COVERED | available=2026-04-17~2026-04-30, required=2026-04-17~2026-05-15
- cn_ga_market_pulse_daily.trade_date: RANGE_NOT_COVERED | available=2026-04-17~2026-04-30, required=2026-04-17~2026-05-15
==================

python scripts/build_cn_mainline_strength_daily.py --start 2026-05-01 --end 2026-05-16 --min-rows 1 --db-host localhost --db-port 3306 --db-user cn_opr_red --db-name cn_market_red --db-password sec_Bobo123

python scripts/build_ga_mainline_radar_daily.py --start 2026-05-01 --end 2026-05-16 --replace --chunk-months 1 --db-host localhost --db-port 3306 --db-user cn_opr_red --db-name cn_market_red --db-password sec_Bobo123

python scripts/build_ga_market_pulse_daily.py --start 2026-05-01 --end 2026-05-16 --replace --chunk-months 3 --db-host localhost --db-port 3306 --db-user cn_opr_red --db-name cn_market_red --db-password sec_Bobo123



[2026-05-17 13:44:10] [Runner] (INFO) [V8] skip build_v7_v8_crosswalk_latest.py because cn_v7_v8_industry_crosswalk is empty or missing ????




No tasks in progress


我的生产db坏了，要全部重建，怎样重建D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2\docs\V8_TABLES_DAILY_AND_BACKFILL_RUNBOOK.md 的tables data ? 
2:26 PM

我先看一下这份 runbook 和仓库里现有的重建脚本/命令，把可执行的重建步骤整理成一套适合生产库全量重建的流程。

Ran 1 command
你要直接去DB里比对， 在这部分是重建及前置条件的数据 又怎重建，它依赖  cn_local_industry_map_hist for 85xxxx.SI
2:28 PM


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
python scripts\build_local_industry_proxy_daily.py 
  --start 2000-01-04 ^
  --end 2026-05-15 ^
  --resume ^
  --workers 4

python scripts\build_tushare_sw_replacement_sources.py --start 2000-01-04 --end 2026-05-15 --srcs  SW2021 SW2014 --levels L1 --replace-master  --replace-members  --output-dir reports\analysis\tushare_sw_replacement_sources_full_rebuild

python scripts\build_v7_v8_industry_crosswalk.py --start 2000-01-04  --end 2026-05-15 --replace   --srcs SW2021 SW2014  --source-mode db  --output-dir reports\analysis\sw_v7_v8_crosswalk_full_rebuild

python scripts\build_v7_v8_crosswalk_latest.py   --replace   --output-dir reports\analysis\v7_v8_crosswalk_latest


python scripts\build_stock_v8_to_v7_sw_map_latest.py  --replace   --output-dir reports\analysis\stock_v8_to_v7_sw_map_latest


先验收这 3 条 SQL

SELECT COUNT(*) AS cnt, COUNT(DISTINCT industry_id) AS industry_cnt
FROM cn_local_industry_map_hist
WHERE industry_id LIKE '85%.SI'; --> 
cnt |industry_cnt|
----+------------+
7101|         274|

SELECT COUNT(*) AS cnt, MIN(effective_date) AS min_d, MAX(effective_date) AS max_d
FROM cn_v7_v8_industry_crosswalk;
cnt   |min_d     |max_d     |
------+----------+----------+
155830|2010-01-29|2026-05-15|

SELECT src, COUNT(*) AS cnt, MAX(asof_date) AS asof_d
FROM cn_v7_v8_industry_crosswalk_latest
GROUP BY src;
src   |cnt|asof_d    |
------+---+----------+
SW2014|274|2026-05-15|
SW2021|274|2026-05-15|


补一句风险提示

我还查到 cn_board_industry_master 里目前 没有 85xxxx.SI 主数据，所以第一步虽然能把 85xxxx.SI membership 重建出来，但部分 industry_name 可能先回落成代码值。如果你这次要求“先恢复可跑、可对齐”，上面这套足够；如果你还要求 85xxxx.SI 名称也完整，我下一步可以继续帮你把 85 系列名字来源补齐。

如果你愿意，我可以下一步直接替你在库里执行这整套重建，并一边跑一边把每一步的实际行数回报给你。



2:37 PM
另一个问题是这些表的数据的下游历史也要重建， 怎建？
2:43 PM


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



2:44 PM
D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2\docs\V8_TASK_SPLIT_RUNBOOK_20260515.md 根据这个文档， 全面整理新文档，V7数据 或全空DB上 进行全部历史数据（2010-2016）的数据顺序 或命令， 除了 v7_v8_industry_crosswalk 外可能还有别的，需要补上 
2:50 PM


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


2:56 PM
在当前的数据的环境中，需要补什么？ 我已经回补了80%, 
3:01 PM

不看库里的实际状态、只按当前仓库定义来判断的话，你“已经回补了 80%”之后，最可能还差的不是原始行情，而是兼容映射链和下游派生链。

set PYTHONPATH=D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2 && python D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2\app\tools\backfill_cn_stock_daily_price_baostock_hist_from_tushare.py --symbol 603014 --start 20250101 --end 20260518 --replace-existing-symbol 2>&1

python D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2\app\tools\backfill_cn_stock_daily_price_baostock_hist_from_tushare.py ^
  --symbol 603014 ^
  --start 20250101 ^
  --end 20260518 ^
  --replace-existing-symbol ^
  --workload-mode all-universe

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




python scripts/build_local_industry_map_hist.py --start 2010-01-04 --end 2016-05-15 --level L1 --src SW2021 --resume -->  [ERROR] - cn_board_member_map_d.trade_date: available=2010-01-04~2016-05-13, 但db 有 至 2026-05-15的数据
6:17 PM


Worked for 56s
根因就是这个，不是你的数据真的缺到 2016-05-15。

2016-05-15 是周日，而 cn_board_member_map_d.trade_date 是交易日表，所以区间内最后一天自然是 2016-05-13（周五）。原来的 build_local_industry_map_hist.py 审计逻辑把你传入的 --end 2016-05-15 当成“必须存在的上限”，因此误报了。

我已经把脚本改好了，逻辑变成：

先解析请求区间的 end
再自动取 end 当天或之前的实际可用交易日终点
审计时用这个 effective_end 去比 cn_board_member_map_d
也就是说，像这次这种：

请求结束日：2016-05-15
实际交易终点：2016-05-13
现在会按 2016-05-13 审计，不会再误拦。语法检查也已经过了。

你现在直接重跑原命令即可：

python scripts/build_local_industry_map_hist.py --start 2010-01-04 --end 2016-05-15 --level L1 --src SW2021 --resume
如果你不想依赖脚本自动处理，也可以手动把结束日改成交易日：

python scripts/build_local_industry_map_hist.py --start 2010-01-04 --end 2016-05-13 --level L1 --src SW2021 --resume
这次我改的是：

build_local_industry_map_hist.py
如果你愿意，我下一步可以顺手把 build_local_industry_proxy_daily.py 这类类似审计也统一改成同样的“非交易日容忍”逻辑，避免你后面继续撞同类问题。


Edited build_local_industry_map_hist.py
+46
-3
Undo
Review
Details


6:18 PM
go, 另一个问题是 这些cn_local_industry_master
cn_local_industry_map_hist
cn_local_industry_proxy_daily
cn_ts_sw_industry_master
cn_ts_sw_industry_member_hist
cn_v7_v8_industry_crosswalk  在 未来的dauly, weekly, monthly 怎样更新？
8:24 PM


Worked for 1m 18s
我已经把两个脚本的“非交易日误报”修掉并过了语法检查：

build_local_industry_map_hist.py
build_local_industry_proxy_daily.py
关于你列的这 6 张表，按当前代码现状，它们并没有完整接进标准 daily / weekly / monthly 主链。当前主链里真正自动做的，只有：

weekly_finalize 会刷 cn_v7_v8_industry_crosswalk_latest
daily/monthly derived alpha 在开关打开时也会刷 crosswalk_latest
但 cn_v7_v8_industry_crosswalk 历史表本身不会自动建
cn_local_industry_* 和 cn_ts_sw_industry_* 也没有接入标准日周月任务
所以这 6 张表，建议你按下面的节奏维护。

建议节奏

cn_local_industry_master

低频表。
建议：monthly 或申万分类有变化时手动刷新。
命令：python scripts/build_sw_industry_master.py --start ... --end ... --src SW2021
cn_local_industry_map_hist

依赖 cn_board_member_map_d，本质上属于 reference/mapping 层。
建议：weekly 刷新。
时机：weekly_refresh / board 完成后再跑。
命令：python scripts/build_local_industry_map_hist.py --start ... --end ... --level L1 --src SW2021 --resume
cn_local_industry_proxy_daily

依赖 map_hist + stock_daily_price + stock_daily_basic。
建议：至少 weekly，如果你后续策略真依赖它，最好 daily 跑最近 5-20 个交易日。
命令：python scripts/build_local_industry_proxy_daily.py --start ... --end ... --resume --workers 4
cn_ts_sw_industry_master

低频官方主数据。
建议：monthly 全量刷新一次就够。
命令由 build_tushare_sw_replacement_sources.py 一起覆盖。
cn_ts_sw_industry_member_hist

官方成员历史，属于 reference history。
建议：weekly 增量刷新，monthly 做一次较宽窗口补齐。
命令：python scripts/build_tushare_sw_replacement_sources.py --start ... --end ... --srcs SW2021 SW2014 --levels L1 --replace-members ...
cn_v7_v8_industry_crosswalk

历史桥接表。
建议：weekly 或 monthly 重建最近窗口，不建议每天全量重跑。
注意：当前自动链只刷 latest，不刷这张历史表本体。
命令：python scripts/build_v7_v8_industry_crosswalk.py --start ... --end ... --replace --srcs SW2021 SW2014 --source-mode db ...
我建议的未来运维策略

daily

不刷 cn_local_industry_master
不刷 cn_ts_sw_industry_master
不刷 cn_v7_v8_industry_crosswalk 历史表
可选刷：cn_local_industry_proxy_daily 最近 5-20 个交易日
可选刷：cn_v7_v8_industry_crosswalk_latest
weekly

先跑 weekly.bat
再补：
cn_local_industry_map_hist
cn_local_industry_proxy_daily
cn_ts_sw_industry_member_hist
cn_v7_v8_industry_crosswalk
cn_v7_v8_industry_crosswalk_latest
cn_stock_v8_to_v7_sw_map_latest
monthly

刷：
cn_local_industry_master
cn_ts_sw_industry_master
再对最近较宽窗口重跑：
cn_ts_sw_industry_member_hist
cn_v7_v8_industry_crosswalk
crosswalk_latest / stock_v8_to_v7_sw_map_latest
最实用的落地方案

你可以把这 6 张表单独当成一个“reference compatibility”链，放在 weekly 后面跑，而不是塞进 daily 主链：

python scripts/build_local_industry_map_hist.py --start 2026-01-01 --end 2026-05-15 --level L1 --src SW2021 --resume
python scripts/build_local_industry_proxy_daily.py --start 2026-01-01 --end 2026-05-15 --resume --workers 4
python scripts/build_tushare_sw_replacement_sources.py --start 2026-01-01 --end 2026-05-15 --srcs SW2021 SW2014 --levels L1 --replace-members --output-dir reports/analysis/tushare_sw_replacement_sources_weekly
python scripts/build_v7_v8_industry_crosswalk.py --start 2026-01-01 --end 2026-05-15 --replace --srcs SW2021 SW2014 --source-mode db --output-dir reports/analysis/sw_v7_v8_crosswalk_weekly
python scripts/build_v7_v8_crosswalk_latest.py --replace
python scripts/build_stock_v8_to_v7_sw_map_latest.py --replace
如果你愿意，我下一步可以直接帮你再写一个 weekly_reference_compatibility.bat，把这 6 张表的周更链固化成一键命令。


Edited build_local_industry_proxy_daily.py
+28
-4
Undo
Review
Details


8:26 PM
如果你要，我下一步可以继续给你做一个 monthly_reference_compatibility.bat，把“月更宽窗口”也单独固化出来。


V8_HISTORICAL_BACKFILL_2010_2016_INCREMENTAL_RUNBOOK.md
Document · MD
Open

Edited 3 files
+329
-11
Undo
Review
weekly_reference_compatibility.bat
docs/V8_HISTORICAL_BACKFILL_2010_2016_INCREMENTAL_RUNBOOK.md
docs/V8_HISTORICAL_BACKFILL_2010_2016_INCREMENTAL_RUNBOOK.md


8:29 PM
go
8:30 PM



Worked for 27s
已经补好了。

新增月更宽窗口入口：
monthly_reference_compatibility.bat

它会复用：
weekly_reference_compatibility.bat

并默认打开这些月更选项：

V8_REFCOMP_REFRESH_LOCAL_MASTER=1
V8_REFCOMP_REPLACE_TS_MASTER=1
V8_REFCOMP_REPLACE_TS_MEMBERS=1
V8_REFCOMP_REBUILD_STOCK_MAP_LATEST=1
现在你的维护入口可以这样用：

周更：

weekly.bat --start-date 2026-05-01 --end-date 2026-05-15 --refresh
weekly_reference_compatibility.bat --start-date 2026-05-01 --end-date 2026-05-15
月更：

monthly.bat --start-date 2026-01-01 --end-date 2026-05-15 --refresh
monthly_reference_compatibility.bat --start-date 2026-01-01 --end-date 2026-05-15
我也把这个入口补进了：
docs/V8_HISTORICAL_BACKFILL_2010_2016_INCREMENTAL_RUNBOOK.md

如果你愿意，下一步我可以继续帮你把这两条兼容链再合成一个总入口，比如 weekly_full_with_refcompat.bat 和 monthly_full_with_refcompat.bat。


V8_HISTORICAL_BACKFILL_2010_2016_INCREMENTAL_RUNBOOK.md

weekly_reference_compatibility.bat  --start-date 2000-01-04 --end-date 2026-05-16
weekly_reference_compatibility
monthly_reference_compatibility

:: Run with defaults (weekly: last 30 days, monthly: last 50 days)
weekly_reference_compatibility.bat
monthly_reference_compatibility.bat

:: Explicit date range still works
weekly_reference_compatibility.bat --start-date 2026-01-01 --end-date 2026-05-15
monthly_reference_compatibility.bat --start-date 2026-01-01 --end-date 2026-05-15



===============

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
set V8_DERIVED_FOUNDATION_STRICT=0
set V8_BACKFILL_START=20100101
set V8_BACKFILL_END=20101231
python runner.py --flag tu --tasks v8_backfill --asof 20101231 《====



Rerun --
python scripts\backtest_adaptive.py --config config\config_r28_e123_e160_e170_p8_decision_support_cn_market_red.yaml  --year 2025   --run-id P10A_ZERO_BASE_2025_20260518


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


build_sw_industry_member_hist.py


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



可以补，而且这张表不能只靠 weekly/daily 常规刷新补回来。
如果 cn_board_industry_member_hist 已经丢了 98-99%，对 DECISION_SUPPORT 来说，正确做法是：

先重建 cn_board_industry_member_hist 历史
再重建 cn_board_member_map_d
再重跑行业轮动下游历史
关键点是：你这个场景应该补 SW L1，不是只跑默认的 board 周刷。仓库里默认 BoardMembershipRefreshTask 走的是 SW2021 L3 当日快照增量，不适合恢复 2010 以来的一级行业历史。相关说明在 docs/board_membership_refresh_playbook.md 和 docs/sw_l1_industry_landing_plan.md。

最直接的补法是跑这个工具：
app/tools/backfill_sw_industry_history_from_tushare.py

如果你的 DECISION_SUPPORT 历史从 2010 开始，建议这样跑：

set TUSHARE_TOKEN=YOUR_TOKEN
python -m app.tools.backfill_sw_industry_history_from_tushare ^
  --start 2010-01-04 ^
  --end 2026-05-18 ^
  --src SW2021 ^
  --level L1 ^
  --master-source TUSHARE_SW2021_L1 ^
  --member-source tushare_sw_l1 ^
  --map-chunk-years 1
这条命令会做三件事：

go 
12:18 PM

下面给你一份 MySQL 客户端里直接可复制执行的 SQL/SP 恢复清单。
假设你已经把 cn_board_industry_member_hist 补好了，接下来从 2010-01-04 到 2016-12-31 重建行业轮动下游。

1. 先确保 rotation 核心 DDL/SP 在库里

USE cn_market;

SOURCE docs/DDL/cn_market.sp_refresh_sector_eod_hist.sql;
SOURCE docs/DDL/cn_market.cn_sector_rotation_transition_v.sql;
SOURCE docs/DDL/cn_market.sp_build_sector_rotation_ranked_by_date.sql;
SOURCE docs/DDL/cn_market.sp_build_sector_rotation_signal_by_date.sql;
SOURCE docs/DDL/cn_market.sp_backfill_rot_bt_from_price.sql;
SOURCE docs/DDL/cn_market.sp_repair_rot_bt_nav.sql;
SOURCE docs/DDL/cn_market.sp_refresh_rotation_snap_all.sql;
SOURCE docs/DDL/cn_market.sp_rotation_daily_refresh.sql;
2. 如果 cn_board_member_map_d 还要再重建一遍

CALL sp_build_board_member_map('2010-01-04', '2016-12-31');
3. 清掉旧的 rotation 三张历史表区间

DELETE FROM cn_sector_eod_hist_t
WHERE trade_date BETWEEN '2010-01-04' AND '2016-12-31';

DELETE FROM cn_sector_rotation_ranked_t
WHERE trade_date BETWEEN '2010-01-04' AND '2016-12-31';

DELETE FROM cn_sector_rotation_signal_t
WHERE signal_date BETWEEN '2010-01-04' AND '2016-12-31';
4. 建一个临时过程，按交易日重跑 eod/ranked/signal

DROP PROCEDURE IF EXISTS sp_rebuild_rotation_history_window;
DELIMITER $$

CREATE PROCEDURE sp_rebuild_rotation_history_window(
    IN p_start DATE,
    IN p_end DATE,
    IN p_top_pct DECIMAL(10,4),
    IN p_breadth_min DECIMAL(10,4)
)
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE v_td DATE;

    DECLARE cur CURSOR FOR
        SELECT DISTINCT TRADE_DATE
        FROM cn_stock_daily_price
        WHERE TRADE_DATE BETWEEN p_start AND p_end
        ORDER BY TRADE_DATE;

    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

    CALL sp_refresh_sector_eod_hist(p_start, p_end, p_top_pct, p_breadth_min);

    OPEN cur;

    read_loop: LOOP
        FETCH cur INTO v_td;
        IF done = 1 THEN
            LEAVE read_loop;
        END IF;

        CALL SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE(v_td);
        CALL SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE(v_td);
    END LOOP;

    CLOSE cur;
END$$

DELIMITER ;
5. 执行历史重建

CALL sp_rebuild_rotation_history_window('2010-01-04', '2016-12-31', 0.30, 0.60);
6. 如果你还要回测净值表

CALL SP_BACKFILL_ROT_BT_FROM_PRICE('SR_LIVE_DEFAULT', '2016-12-31', 1);
这会补：

cn_sector_rot_bt_daily_t
7. 如果你还要最终快照表，只刷最后一天

CALL SP_REFRESH_ROTATION_SNAP_ALL('SR_LIVE_DEFAULT', '2016-12-31', 1);
这会刷：

cn_rotation_entry_snap_t
cn_rotation_holding_snap_t
cn_rotation_exit_snap_t
8. 验收 SQL

SELECT COUNT(DISTINCT trade_date) AS eod_days
FROM cn_sector_eod_hist_t
WHERE trade_date BETWEEN '2010-01-04' AND '2016-12-31';

SELECT COUNT(DISTINCT trade_date) AS ranked_days
FROM cn_sector_rotation_ranked_t
WHERE trade_date BETWEEN '2010-01-04' AND '2016-12-31';

SELECT COUNT(DISTINCT signal_date) AS signal_days
FROM cn_sector_rotation_signal_t
WHERE signal_date BETWEEN '2010-01-04' AND '2016-12-31';
SELECT COUNT(*) AS bt_rows
FROM cn_sector_rot_bt_daily_t
WHERE run_id = 'SR_LIVE_DEFAULT'
  AND trade_date BETWEEN '2010-01-04' AND '2016-12-31';
如果你愿意，我下一条可以继续给你一份 “按年执行版 SQL 清单”，把 2010 到 2016 分成年块，出错时更好重跑。



12:18 PM
go
12:18 PM

下面给你一份 按年执行版 SQL 清单。
这版更适合生产恢复，某一年失败时只重跑那一年。

1. 先确保核心 DDL / SP 都在

USE cn_market;

SOURCE docs/DDL/cn_market.sp_refresh_sector_eod_hist.sql;
SOURCE docs/DDL/cn_market.cn_sector_rotation_transition_v.sql;
SOURCE docs/DDL/cn_market.sp_build_sector_rotation_ranked_by_date.sql;
SOURCE docs/DDL/cn_market.sp_build_sector_rotation_signal_by_date.sql;
SOURCE docs/DDL/cn_market.sp_backfill_rot_bt_from_price.sql;
SOURCE docs/DDL/cn_market.sp_repair_rot_bt_nav.sql;
SOURCE docs/DDL/cn_market.sp_refresh_rotation_snap_all.sql;
SOURCE docs/DDL/cn_market.sp_rotation_daily_refresh.sql;
2. 建一个“按年重跑”的临时过程

DROP PROCEDURE IF EXISTS sp_rebuild_rotation_year;
DELIMITER $$



CREATE PROCEDURE sp_rebuild_rotation_year(
    IN p_start DATE,
    IN p_end DATE,
    IN p_top_pct DECIMAL(10,4),
    IN p_breadth_min DECIMAL(10,4)
)
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE v_td DATE;

    DECLARE cur CURSOR FOR
        SELECT DISTINCT TRADE_DATE
        FROM cn_stock_daily_price
        WHERE TRADE_DATE BETWEEN p_start AND p_end
        ORDER BY TRADE_DATE;

    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

    DELETE FROM cn_sector_eod_hist_t
    WHERE trade_date BETWEEN p_start AND p_end;

    DELETE FROM cn_sector_rotation_ranked_t
    WHERE trade_date BETWEEN p_start AND p_end;

    DELETE FROM cn_sector_rotation_signal_t
    WHERE signal_date BETWEEN p_start AND p_end;

    CALL sp_refresh_sector_eod_hist(p_start, p_end, p_top_pct, p_breadth_min);

    OPEN cur;

    read_loop: LOOP
        FETCH cur INTO v_td;
        IF done = 1 THEN
            LEAVE read_loop;
        END IF;

        CALL SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE(v_td);
        CALL SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE(v_td);
    END LOOP;

    CLOSE cur;
END$$

DELIMITER ;
3. 逐年执行 2010-2016

CALL sp_rebuild_rotation_year('2010-01-04', '2010-12-31', 0.30, 0.60);
CALL sp_rebuild_rotation_year('2011-01-01', '2011-12-31', 0.30, 0.60);
CALL sp_rebuild_rotation_year('2012-01-01', '2012-12-31', 0.30, 0.60);
CALL sp_rebuild_rotation_year('2013-01-01', '2013-12-31', 0.30, 0.60);
CALL sp_rebuild_rotation_year('2014-01-01', '2014-12-31', 0.30, 0.60);
CALL sp_rebuild_rotation_year('2015-01-01', '2015-12-31', 0.30, 0.60);
CALL sp_rebuild_rotation_year('2016-01-01', '2016-12-31', 0.30, 0.60);
4. 如果还要回测净值，最后统一补一次

CALL SP_BACKFILL_ROT_BT_FROM_PRICE('SR_LIVE_DEFAULT', '2016-12-31', 1);
5. 如果还要最终快照，最后刷一次末日

CALL SP_REFRESH_ROTATION_SNAP_ALL('SR_LIVE_DEFAULT', '2016-12-31', 1);
6. 每年跑完后的验收 SQL

SELECT COUNT(DISTINCT trade_date) AS eod_days
FROM cn_sector_eod_hist_t
WHERE trade_date BETWEEN '2010-01-04' AND '2010-12-31';

SELECT COUNT(DISTINCT trade_date) AS ranked_days
FROM cn_sector_rotation_ranked_t
WHERE trade_date BETWEEN '2010-01-04' AND '2010-12-31';

SELECT COUNT(DISTINCT signal_date) AS signal_days
FROM cn_sector_rotation_signal_t
WHERE signal_date BETWEEN '2010-01-04' AND '2010-12-31';
把年份换掉就行。

最实用的执行顺序

先补 cn_board_industry_member_hist
再补 cn_board_member_map_d
再逐年 CALL sp_rebuild_rotation_year(...)
最后 SP_BACKFILL_ROT_BT_FROM_PRICE
最后 SP_REFRESH_ROTATION_SNAP_ALL
如果你愿意，我下一条可以继续给你一份 “恢复 DECISION_SUPPORT 最终验收 SQL 清单”，专门看行业板块映射、rotation 三表、回测表是不是已经回到正常口径。



12:18 PM
go
12:19 PM

下面给你一份 DECISION_SUPPORT 恢复后的最终验收 SQL 清单。
这套只看“口径是否回来了”，不依赖我去看你库里的实际现状。

建议按 4 组验收：

板块成员历史
日展开映射
rotation 三张核心表
回测/快照结果
1. 验收 cn_board_industry_member_hist

先看一级行业历史是否恢复到合理量级：

SELECT
    COUNT(*) AS hist_rows,
    COUNT(DISTINCT board_id) AS board_cnt,
    COUNT(DISTINCT symbol) AS symbol_cnt,
    MIN(valid_from) AS min_valid_from,
    MAX(COALESCE(valid_to, DATE('9999-12-31'))) AS max_valid_to
FROM cn_board_industry_member_hist
WHERE board_id LIKE '801%.SI';
看 source 是否正确落在你这次补的那条链上：

SELECT
    source,
    COUNT(*) AS row_cnt,
    COUNT(DISTINCT board_id) AS board_cnt
FROM cn_board_industry_member_hist
WHERE board_id LIKE '801%.SI'
GROUP BY source
ORDER BY row_cnt DESC;
抽查是否仍然只剩“几十行”这种坏状态：

SELECT
    board_id,
    COUNT(*) AS row_cnt,
    MIN(valid_from) AS min_valid_from,
    MAX(COALESCE(valid_to, DATE('9999-12-31'))) AS max_valid_to
FROM cn_board_industry_member_hist
WHERE board_id LIKE '801%.SI'
GROUP BY board_id
ORDER BY row_cnt DESC, board_id;
2. 验收 cn_board_member_map_d

先看 2010-2016 交易日覆盖：

SELECT
    MIN(trade_date) AS min_d,
    MAX(trade_date) AS max_d,
    COUNT(DISTINCT trade_date) AS d_cnt,
    COUNT(*) AS row_cnt
FROM cn_board_member_map_d
WHERE trade_date BETWEEN '2010-01-04' AND '2016-12-31'
  AND sector_type = 'INDUSTRY';
看一级行业映射是否真的存在：

SELECT
    COUNT(*) AS row_cnt,
    COUNT(DISTINCT sector_id) AS sector_cnt,
    COUNT(DISTINCT symbol) AS symbol_cnt
FROM cn_board_member_map_d
WHERE trade_date BETWEEN '2010-01-04' AND '2016-12-31'
  AND sector_type = 'INDUSTRY'
  AND sector_id LIKE '801%.SI';
按年看有没有断层：

SELECT
    YEAR(trade_date) AS yr,
    COUNT(DISTINCT trade_date) AS d_cnt,
    COUNT(*) AS row_cnt
FROM cn_board_member_map_d
WHERE trade_date BETWEEN '2010-01-04' AND '2016-12-31'
  AND sector_type = 'INDUSTRY'
  AND sector_id LIKE '801%.SI'
GROUP BY YEAR(trade_date)
ORDER BY yr;
3. 验收 rotation 三张核心表

先看三表日期覆盖是否一致：

WITH p AS (
  SELECT DISTINCT TRADE_DATE d
  FROM cn_stock_daily_price
  WHERE TRADE_DATE BETWEEN '2010-01-04' AND '2016-12-31'
),
e AS (
  SELECT DISTINCT trade_date d
  FROM cn_sector_eod_hist_t
  WHERE trade_date BETWEEN '2010-01-04' AND '2016-12-31'
),
r AS (
  SELECT DISTINCT trade_date d
  FROM cn_sector_rotation_ranked_t
  WHERE trade_date BETWEEN '2010-01-04' AND '2016-12-31'
),
s AS (
  SELECT DISTINCT signal_date d
  FROM cn_sector_rotation_signal_t
  WHERE signal_date BETWEEN '2010-01-04' AND '2016-12-31'
)
SELECT
  (SELECT COUNT(*) FROM p) AS price_days,
  (SELECT COUNT(*) FROM e) AS eod_days,
  (SELECT COUNT(*) FROM r) AS ranked_days,
  (SELECT COUNT(*) FROM s) AS signal_days;
看三表按年是否齐：

SELECT 'eod' AS src, YEAR(trade_date) AS yr, COUNT(*) AS row_cnt
FROM cn_sector_eod_hist_t
WHERE trade_date BETWEEN '2010-01-04' AND '2016-12-31'
GROUP BY YEAR(trade_date)

UNION ALL

SELECT 'ranked', YEAR(trade_date), COUNT(*)
FROM cn_sector_rotation_ranked_t
WHERE trade_date BETWEEN '2010-01-04' AND '2016-12-31'
GROUP BY YEAR(trade_date)

UNION ALL

SELECT 'signal', YEAR(signal_date), COUNT(*)
FROM cn_sector_rotation_signal_t
WHERE signal_date BETWEEN '2010-01-04' AND '2016-12-31'
GROUP BY YEAR(signal_date)

ORDER BY src, yr;
抽查 rotation 是否真在用行业而不是空映射：

SELECT
    sector_type,
    COUNT(*) AS row_cnt,
    COUNT(DISTINCT sector_id) AS sector_cnt,
    MIN(trade_date) AS min_d,
    MAX(trade_date) AS max_d
FROM cn_sector_rotation_ranked_t
WHERE trade_date BETWEEN '2010-01-04' AND '2016-12-31'
GROUP BY sector_type;
看信号表是否正常出信号：

SELECT
    action,
    COUNT(*) AS row_cnt,
    COUNT(DISTINCT signal_date) AS d_cnt
FROM cn_sector_rotation_signal_t
WHERE signal_date BETWEEN '2010-01-04' AND '2016-12-31'
GROUP BY action
ORDER BY row_cnt DESC;
4. 验收回测 / 快照

如果你补了回测表：

SELECT
    run_id,
    MIN(trade_date) AS min_d,
    MAX(trade_date) AS max_d,
    COUNT(*) AS row_cnt,
    MAX(nav) AS max_nav,
    MIN(nav) AS min_nav
FROM cn_sector_rot_bt_daily_t
WHERE trade_date BETWEEN '2010-01-04' AND '2016-12-31'
GROUP BY run_id
ORDER BY row_cnt DESC;
看持仓表有没有正常记录：

SELECT
    run_id,
    COUNT(*) AS row_cnt,
    COUNT(DISTINCT trade_date) AS d_cnt
FROM cn_sector_rot_pos_daily_t
WHERE trade_date BETWEEN '2010-01-04' AND '2016-12-31'
GROUP BY run_id
ORDER BY row_cnt DESC;
如果你刷了最终快照，只看最后一天：

SELECT COUNT(*) AS entry_cnt
FROM cn_rotation_entry_snap_t
WHERE trade_date = '2016-12-31';

SELECT COUNT(*) AS holding_cnt
FROM cn_rotation_holding_snap_t
WHERE trade_date = '2016-12-31';

SELECT COUNT(*) AS exit_cnt
FROM cn_rotation_exit_snap_t
WHERE trade_date = '2016-12-31';
最关键的判断标准

如果下面 4 条都成立，基本可以认为 DECISION_SUPPORT 的行业轮动底层已恢复：

cn_board_industry_member_hist 的 801%.SI 不再是几十行，而是恢复到成体系的历史量级。
cn_board_member_map_d 在 2010-01-04 到 2016-12-31 对 801%.SI 有连续交易日覆盖。
cn_sector_eod_hist_t / ranked / signal 的日期覆盖接近一致。
signal 和 bt 不再是空表或极低行数。
如果你愿意，我下一条可以继续给你一份 “异常结果判读表”，比如：

hist_rows 正常但 map 很少
map 正常但 ranked/signal 很少
signal 正常但 bt 几乎为空
分别意味着哪一步还没补对。


12:19 PM
go
12:20 PM

下面这份是 异常结果判读表。
你跑完验收 SQL 后，可以直接按现象定位哪一步还没补对。

1. cn_board_industry_member_hist 很少，只有几十行到几百行

最可能原因：

还没真正跑 SW2021 L1 历史回补
跑成了默认的 SW L3
用了错误的 source
加了 --keep-existing-member-source，坏数据没被清掉
怎么判断：

board_id like '801%.SI' 行数很少
source 里没有 tushare_sw_l1
只有 850xxx.SI 这类 L3 或别的 taxonomy
怎么补：

重跑 python -m app.tools.backfill_sw_industry_history_from_tushare --level L1
不要加 --keep-existing-member-source
2. cn_board_industry_member_hist 正常，但 cn_board_member_map_d 很少

最可能原因：

sp_build_board_member_map 没跑
跑了，但时间范围太短
cn_stock_daily_price 交易日覆盖不足，导致 map 展不开
怎么判断：

hist_rows 很多，但 map 在 2010-2016 的 d_cnt 很少
map 的 min_d/max_d 不覆盖目标区间
price_days 明显大于 map_days
怎么补：

CALL sp_build_board_member_map('2010-01-04', '2016-12-31');
如果还是不够，就先检查 cn_stock_daily_price 的交易日覆盖。

3. map 正常，但 cn_sector_eod_hist_t 很少或为空

最可能原因：

sp_refresh_sector_eod_hist 没跑
相关 DDL/SP/view 缺失
cn_sector_rotation_transition_v 没建好
sector eod 依赖的源表不齐
怎么判断：

map 有连续交易日
eod_days 远小于 price_days
执行 rotation 时曾报 1146 或 view/procedure not found
怎么补：

先补 DDL：

SOURCE docs/DDL/cn_market.sp_refresh_sector_eod_hist.sql;
SOURCE docs/DDL/cn_market.cn_sector_rotation_transition_v.sql;
SOURCE docs/DDL/cn_market.sp_build_sector_rotation_ranked_by_date.sql;
SOURCE docs/DDL/cn_market.sp_build_sector_rotation_signal_by_date.sql;
再重跑 rebuild_rotation_three_tables_from_map.py 或对应 SP。

4. eod 正常，但 ranked/signal 很少

最可能原因：

只跑了 sp_refresh_sector_eod_hist，没跑后面的 ranked/signal
SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE
SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE
没执行
transition view 逻辑异常
怎么判断：

eod_days 接近 price_days
ranked_days 或 signal_days 明显偏少
ranked 有数据但 signal 很少，说明 signal 那层没补完
怎么补：

逐日重跑 ranked/signal，或者直接用：

python -m app.tools.rebuild_rotation_three_tables_from_map ^
  --start 2010-01-04 ^
  --end 2016-12-31 ^
  --months-per-chunk 1 ^
  --clear-first 1 ^
  --rank-signal-mode hybrid_base


5. ranked/signal 正常，但 cn_sector_rot_bt_daily_t 几乎为空

最可能原因：

没跑 SP_BACKFILL_ROT_BT_FROM_PRICE
run_id 不对
cn_sector_rot_pos_daily_t 没有持仓数据
只重建了信号，没有重建回测链
怎么判断：

signal 有很多 ENTER/EXIT
bt 很少或为 0
run_id='SR_LIVE_DEFAULT' 下几乎没记录
怎么补：

CALL SP_BACKFILL_ROT_BT_FROM_PRICE('SR_LIVE_DEFAULT', '2016-12-31', 1);
如果还是空，再查 cn_sector_rot_pos_daily_t。

6. bt 有数据，但 nav 很怪，几乎不动或断掉

最可能原因：

bt 表是旧残留，没有正确 force 重建
nav 断点需要修复
价格或信号链中间有断层
怎么判断：

row_cnt 有，但 nav 长期等于 1
中间年份缺很多交易日
max(nav) 和 min(nav) 几乎一样
怎么补：

先强制回补：

CALL SP_BACKFILL_ROT_BT_FROM_PRICE('SR_LIVE_DEFAULT', '2016-12-31', 1);
必要时再跑：

CALL sp_repair_rot_bt_nav('SR_LIVE_DEFAULT');
7. 三张 rotation 表都有数据，但 DECISION_SUPPORT 结果还是明显不对

最可能原因：

报告读的不是这条行业 taxonomy
仍然在用 cn_board_industry_cons / cn_board_concept_cons 这种近端 snapshot
读的是旧 850xxx.SI、Eastmoney BK%，不是 801%.SI
报告缓存或中间物化表没刷新
怎么判断：

rotation 表里 sector_id 类型不对
ranked/signal 里行业名称还是旧命名
报表结果和 cn_sector_rotation_* 明显对不上
怎么补：

确认报告 SQL 用的是 cn_board_member_map_d
确认行业口径是 801%.SI
重刷报表中间表/缓存
8. hist/map/rotation 都正常，但某几年特别少

最可能原因：

年度重跑过程中某一年失败
resume 跳过了坏块
某一年 price 或 map 本身缺口
怎么判断：

按年统计时只有个别年份掉下去
前后年份正常，中间某年异常低
怎么补：

只重跑异常年份即可，比如：

CALL sp_rebuild_rotation_year('2013-01-01', '2013-12-31', 0.30, 0.60);
最实用的排障顺序

永远按这个顺序看：

cn_board_industry_member_hist
cn_board_member_map_d
cn_sector_eod_hist_t
cn_sector_rotation_ranked_t
cn_sector_rotation_signal_t
cn_sector_rot_bt_daily_t
DECISION_SUPPORT 最终报表
也就是：先查映射，再查特征，再查信号，再查回测，最后查报告。
不要一上来盯报告，不然很难定位。

如果你愿意，我下一条可以继续给你一份 “恢复完成后的运维方案”，告诉你以后这几层 daily / weekly / monthly 分别怎么维护，避免再掉坑。



12:20 PM
go
12:20 PM

下面给你一份 恢复完成后的运维方案，目标是以后不再把 DECISION_SUPPORT 的行业轮动底层跑坏。

核心原则就一句：

历史成员表周更，日映射日补，rotation 每日增量，历史修复按年或按月分块。

1. Daily：每天要跑什么

每天收盘后，至少保证这几层：

cn_stock_daily_price
cn_index_daily_price
cn_stock_daily_basic
cn_sw_industry_daily
cn_board_member_map_d
rotation 当日三张表：
cn_sector_eod_hist_t
cn_sector_rotation_ranked_t
cn_sector_rotation_signal_t
如果你们继续走现有主链，最稳的是：

python runner.py --tasks stock,index,stock_basic,sw_industry --asof latest
python runner.py --tasks rotation --asof latest
如果你们日更里已经封在统一入口，就保持现有 daily，但要确保 rotation 没被关掉。

Daily 必开关注意

ROTATION_REFRESH_MODE=sp
不要让 daily 报告去读 cn_board_industry_cons
报告和 rotation 一律基于 cn_board_member_map_d
2. Weekly：每周要跑什么

每周最重要的是纠偏行业成员历史。
因为 cn_board_industry_member_hist 是你这次出问题的根。

建议每周固定做：

增量刷新行业成员历史
刷 cn_board_member_map_d 最近窗口
刷 latest / leader / universe 这类周纠偏层
最稳做法：

python runner.py --tasks board --asof latest
但要注意：

这个默认更偏“当前增量快照”
如果你们 DECISION_SUPPORT 主要看 SW L1
那就建议额外固定做一次 L1 历史补齐窗口
例如最近一年窗口：

set TUSHARE_TOKEN=YOUR_TOKEN
python -m app.tools.backfill_sw_industry_history_from_tushare ^
  --start 2025-01-01 ^
  --end 2026-05-18 ^
  --src SW2021 ^
  --level L1 ^
  --master-source TUSHARE_SW2021_L1 ^
  --member-source tushare_sw_l1 ^
  --map-chunk-years 1 ^
  --keep-existing-member-source
周更用 --keep-existing-member-source 就合理，因为这时是补新窗口，不是灾难恢复。

3. Monthly：每月要跑什么

每月做一次宽窗口校正，防止周更漏数、源端修订、成员历史断层。

推荐每月做：

宽窗口补 cn_board_industry_member_hist
宽窗口重建 cn_board_member_map_d
宽窗口重建 rotation 三表
如需要，补回测表
推荐窗口：最近 12-24 个月。

例如：

set TUSHARE_TOKEN=YOUR_TOKEN
python -m app.tools.backfill_sw_industry_history_from_tushare ^
  --start 2024-01-01 ^
  --end 2026-05-18 ^
  --src SW2021 ^
  --level L1 ^
  --master-source TUSHARE_SW2021_L1 ^
  --member-source tushare_sw_l1 ^
  --map-chunk-years 1 ^
  --keep-existing-member-source
再跑：

python -m app.tools.rebuild_rotation_three_tables_from_map ^
  --start 2024-01-01 ^
  --end 2026-05-18 ^
  --months-per-chunk 1 ^
  --clear-first 1 ^
  --rank-signal-mode hybrid_base ^
  --retries 8 ^
  --retry-sleep-sec 3
4. 历史补洞：什么时候用年块重跑

只要出现下面任一情况，就不要靠 daily/weekly 顶过去，要直接按年或按月重跑：

cn_board_industry_member_hist 行数异常下降
cn_board_member_map_d 年度 d_cnt 明显变少
cn_sector_rotation_signal_t 某几年几乎无信号
DECISION_SUPPORT 报表结果突然和过去口径不连续
这时建议：

小问题：按月重跑
大问题：按年重跑
不要全库一把梭，除非全库确实坏了
5. 最小验收监控，建议常驻

以后每周至少看这几条：

SELECT COUNT(*) AS hist_rows, COUNT(DISTINCT board_id) AS board_cnt
FROM cn_board_industry_member_hist
WHERE board_id LIKE '801%.SI';
SELECT YEAR(trade_date) AS yr, COUNT(DISTINCT trade_date) AS d_cnt
FROM cn_board_member_map_d
WHERE sector_type = 'INDUSTRY'
  AND sector_id LIKE '801%.SI'
GROUP BY YEAR(trade_date)
ORDER BY yr;
SELECT
  COUNT(DISTINCT trade_date) AS ranked_days
FROM cn_sector_rotation_ranked_t
WHERE trade_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY);
SELECT
  action, COUNT(*) AS row_cnt
FROM cn_sector_rotation_signal_t
WHERE signal_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
GROUP BY action;
这几条一旦异常，就能比报表先发现问题。

6. 最后的运维建议

最值得固定化的不是“报表跑完没报错”，而是下面这条链的健康度：

cn_board_industry_member_hist
cn_board_member_map_d
cn_sector_eod_hist_t
cn_sector_rotation_ranked_t
cn_sector_rotation_signal_t
只要这 5 层稳定，DECISION_SUPPORT 的行业轮动部分通常就稳定。

如果你愿意，我下一步可以直接帮你整理成一份新的文档，比如：

docs/DECISION_SUPPORT_ROTATION_RECOVERY_AND_OPERATIONS_RUNBOOK.md

把这次恢复、验收、日周月运维全部固化进去。


DECISION_SUPPORT_ROTATION_RECOVERY_AND_OPERATIONS_RUNBOOK.md
Document · MD
Open


12:20 PM
go
12:21 PM



Worked for 58s
已经整理好了：

docs/DECISION_SUPPORT_ROTATION_RECOVERY_AND_OPERATIONS_RUNBOOK.md

这份新文档包含：

cn_board_industry_member_hist 异常丢失后的恢复顺序
SW2021 L1 历史回补命令
cn_board_member_map_d 重建方式
rotation 三张核心表的 Python / SQL 两种重建路径
backtest / snapshot 的补法
验收 SQL
异常结果判读
后续 daily / weekly / monthly 运维建议
我已经快速检查过文档内容和路径，主体没问题。
如果你愿意，下一步我可以继续帮你把旧文档头部加一个跳转说明，避免后面还翻旧 runbook。


DECISION_SUPPORT_ROTATION_RECOVERY_AND_OPERATIONS_RUNBOOK.md
Document · MD
Open

Edited DECISION_SUPPORT_ROTATION_RECOVERY_AND_OPERATIONS_RUNBOOK.md
+573
-0
Undo
Review
Details


12:22 PM





Custom

5.4
Medium


cn_board_industry_master
把成员历史写进 cn_board_industry_member_hist
自动按年调用 sp_build_board_member_map 重建 cn_board_member_map_d
如果你怀疑当前 tushare_sw_l1 这批历史本身已坏，不要加 --keep-existing-member-source；脚本默认会先清掉该 source 再重灌，这正适合这次恢复。
如果你只是补一个新 source、不想动已有同 source 数据，才用 --keep-existing-member-source。

如果你只想补到 2016-12-31，就把 --end 改成那个日期。
如果你要尽可能完整保留更早历史，可以从 2000-01-04 开始。

补完以后，DECISION_SUPPORT 相关下游还要重补，因为它们不会自动跟着源表修复。最少要重跑这层：

cn_sector_eod_hist_t
cn_sector_rotation_ranked_t
cn_sector_rotation_signal_t
如果你有回测结果表，还包括 cn_sector_rot_bt_daily_t / cn_sector_rot_pos_daily_t / cn_rotation_*_snap_t
按仓库文档，顺序是：

重建 cn_board_industry_member_hist
重建 cn_board_member_map_d
再跑 sp_refresh_sector_eod_hist
再跑 ranked/signal/backtest 链
如果你愿意，我下一条可以直接给你一份“只针对 cn_board_industry_member_hist 丢失”的最短恢复清单，按 2010-2026 执行顺序列成 1 到 5 步。


board_membership_refresh_playbook.md
Document · MD
Open

sw_l1_industry_landing_plan.md
Document · MD
Open


12:00 PM



可以补，而且这张表不能只靠 weekly/daily 常规刷新补回来。
如果 cn_board_industry_member_hist 已经丢了 98-99%，对 DECISION_SUPPORT 来说，正确做法是：

先重建 cn_board_industry_member_hist 历史
再重建 cn_board_member_map_d
再重跑行业轮动下游历史
关键点是：你这个场景应该补 SW L1，不是只跑默认的 board 周刷。仓库里默认 BoardMembershipRefreshTask 走的是 SW2021 L3 当日快照增量，不适合恢复 2010 以来的一级行业历史。相关说明在 docs/board_membership_refresh_playbook.md 和 docs/sw_l1_industry_landing_plan.md。

最直接的补法是跑这个工具：
app/tools/backfill_sw_industry_history_from_tushare.py

如果你的 DECISION_SUPPORT 历史从 2010 开始，建议这样跑：

set TUSHARE_TOKEN=YOUR_TOKEN
python -m app.tools.backfill_sw_industry_history_from_tushare ^
  --start 2010-01-04 ^
  --end 2026-05-18 ^
  --src SW2021 ^
  --level L1 ^
  --master-source TUSHARE_SW2021_L1 ^
  --member-source tushare_sw_l1 ^
  --map-chunk-years 1
这条命令会做三件事：

把 SW2021 L1 主数据写进 cn_board_industry_master
把成员历史写进 cn_board_industry_member_hist
自动按年调用 sp_build_board_member_map 重建 cn_board_member_map_d
如果你怀疑当前 tushare_sw_l1 这批历史本身已坏，不要加 --keep-existing-member-source；脚本默认会先清掉该 source 再重灌，这正适合这次恢复。
如果你只是补一个新 source、不想动已有同 source 数据，才用 --keep-existing-member-source。

如果你只想补到 2016-12-31，就把 --end 改成那个日期。
如果你要尽可能完整保留更早历史，可以从 2000-01-04 开始。

补完以后，DECISION_SUPPORT 相关下游还要重补，因为它们不会自动跟着源表修复。最少要重跑这层：

cn_sector_eod_hist_t
cn_sector_rotation_ranked_t
cn_sector_rotation_signal_t
如果你有回测结果表，还包括 cn_sector_rot_bt_daily_t / cn_sector_rot_pos_daily_t / cn_rotation_*_snap_t
按仓库文档，顺序是：

重建 cn_board_industry_member_hist
重建 cn_board_member_map_d
再跑 sp_refresh_sector_eod_hist
再跑 ranked/signal/backtest 链
如果你愿意，我下一条可以直接给你一份“只针对 cn_board_industry_member_hist 丢失”的最短恢复清单，按 2010-2026 执行顺序列成 1 到 5 步。

---

最短恢复清单

重建 cn_board_industry_member_hist 的 SW L1 历史。
set TUSHARE_TOKEN=16acec7cb9250190ecd69818107d7e4307a097235248830225a432fa
python -m app.tools.backfill_sw_industry_history_from_tushare --start 2010-01-04  --end 2026-05-18  --src SW2021 --level L1 --master-source TUSHARE_SW2021_L1  --member-source tushare_sw_l1   --map-chunk-years 1

如果你只想先恢复 2010-2016，把结束日改成 2016-12-31。
python -m app.tools.backfill_sw_industry_history_from_tushare ^
  --start 2010-01-04 ^
  --end 2016-12-31 ^
  --src SW2021 ^
  --level L1 ^
  --master-source TUSHARE_SW2021_L1 ^
  --member-source tushare_sw_l1 ^
  --map-chunk-years 1
这一步跑完后，cn_board_member_map_d 会一起按年重建；如果你想单独再补一遍 map，可再跑：

python scripts/build_board_member_map_full.py --start 2010-01-01 --end 2016-12-31 --resume  --db-host 127.0.0.1 --db-port 3306  --db-user cn_opr_red --db-password sec_Bobo123 --db-name cn_market_red  
  


然后重跑行业轮动下游历史。核心是先重建 sector eod，再重建 ranked/signal/backtest。
如果你们现在线上就是走月度/分块 SP 链，就按现有 rotation 历史链重跑 2010-2016。

最后再重出 DECISION_SUPPORT 报告，不要在第 1 步后立刻比报告，因为下游物化表还没刷新。

最关键的注意点

不要只跑 weekly.bat 或 runner.py --tasks board，那是当期增量，不是历史恢复。
这次要补的是 SW2021 L1，不是默认的 SW L3。
不要加 --keep-existing-member-source，否则坏掉的 tushare_sw_l1 旧数据可能不会被清干净。
如果 DECISION_SUPPORT 只看 2010 起，先补 2010-01-04 到目标结束日就够。
建议你现在就执行的命令


set TUSHARE_TOKEN=16acec7cb9250190ecd69818107d7e4307a097235248830225a432fa
python -m app.tools.backfill_sw_industry_history_from_tushare --start 2010-01-04 --end 2026-05-16   --src SW2021 --level L1 --master-source TUSHARE_SW2021_L1   --member-source tushare_sw_l1 ^
  --map-chunk-years 1
如果你愿意，我下一条可以继续直接给你补一份“行业轮动下游历史重跑清单”，把 cn_sector_eod_hist_t、ranked、signal、bt 这一串也压成最短命令。


2026-01-09 的数据暴涨是因为 AKShare（东方财富）概念板块数据被首次导入到 cn_board_concept_master 表，然后通过 cn_board_member_map_d 的构建脚本合并到了每日板块成员映射中。

具体来说：

2026-01-09：441 个 AKShare/东方财富概念（BK 前缀）被导入 cn_board_concept_master，ASOF_DATE=2026-01-09
2026-04-30：879 个 Tushare 概念（TS 前缀）被导入，ASOF_DATE=2026-04-30
从 2026-01-09 开始，cn_board_member_map_d 的构建脚本同时从两个数据源拉取数据，导致每日行数从 ~14,500 暴涨到 ~76,700
这不是数据缺漏，而是系统在 2026-01-09 增加了 AKShare/东方财富作为新的概念板块数据源。 这是有意的数据源扩展，不是错误。


====
支持多种运行模式：

# 收盘后运行：物化今天的数据
python scripts/daily_materialize_leader_score.py --today

# 物化最近7天（含今天，默认行为）
python scripts/daily_materialize_leader_score.py --lookback 7

# 指定日期范围
python scripts/daily_materialize_leader_score.py --start 2026-05-20 --end 2026-05-21

# 全量历史回填（首次运行）
python scripts/daily_materialize_leader_score.py --backfill

脚本功能：

自动检查 v2 视图是否存在、物化表是否存在
逐日物化，每批提交，可中断恢复
使用 ON DUPLICATE KEY UPDATE，幂等安全
输出物化行数、耗时、分区使用情况



=
cn_stock_quality_score_daily 《===
cn_stock_leader_score_daily
cn_stock_fundamental_daily
cn_ga_stock_role_map_daily
cn_unified_alpha_score_daily



==============
步骤 1：补 local_industry_proxy（独立脚本）
cd /d d:\LHJ\PythonWS\MarketScraper\AsharesScraperV2
python scripts/build_local_industry_proxy_daily.py --start 20260429 --end 20260521

步骤 2：补 derived foundation（角色映射等）
cd /d d:\LHJ\PythonWS\MarketScraper\AsharesScraperV2
set V8_DAILY_AUTO_REPAIR=0
python runner.py --flag tu --tasks v8_daily_derived_foundation --asof 20260521 --days 29

这会覆盖从 2026-04-23 到今天的 foundation 层（cn_ga_stock_role_map_daily 等）。

步骤 3：补 derived mainline（主链）
cd /d d:\LHJ\PythonWS\MarketScraper\AsharesScraperV2
set V8_DAILY_AUTO_REPAIR=0
python runner.py --flag tu --tasks v8_daily_derived_mainline --asof 20260521 --days 29

这会覆盖从 2026-04-23 到今天的所有 mainline 表（strength / radar / pulse / lifecycle）。

步骤 4：补 derived alpha（统一 alpha 分）
cd /d d:\LHJ\PythonWS\MarketScraper\AsharesScraperV2
set V8_DAILY_AUTO_REPAIR=0
python runner.py --flag tu --tasks v8_daily_derived_alpha --asof 20260521 --days 29

或者一步到位：直接跑完整 daily 链（推荐，因为有自愈逻辑）
cd /d d:\LHJ\PythonWS\MarketScraper\AsharesScraperV2
set V8_DAILY_AUTO_REPAIR=1
set V8_DAILY_REPAIR_LOOKBACK_DAYS=30
python runner.py --flag tu --tasks v8_daily --asof 20260521 --days 29

这会自动检测缺失日期并补回所有 raw + derived 表。但注意 daily_spot_update.bat 中默认跳过了 rotation 和 crosswalk，所以 v8_daily 也不会碰那些。






================= 
Update: 2026-05-21:

cd /d d:\LHJ\PythonWS\MarketScraper\AsharesScraperV2
python scripts/daily_materialize_leader_score.py --lookback 30
--start 和 --end 在同一个互斥组里，所以不能同时用


只想单独回补 lifecycle 表（不重新跑 strength/radar/market_pulse）
前提是上面 4 张上游表已有数据，可以直接调用脚本：

scripts/build_local_industry_map_hist.py --start 1990-01-01 --end 2026-05-21 --level L1 --force

python scripts/build_local_industry_proxy_daily.py --start 2026-05-01 --end 2026-05-21 --workers 4

python scripts/build_mainline_lifecycle_daily.py   --start 2026-05-21  --db-host 127.0.0.1 --db-port 3306  --db-user cn_opr_red --db-name cn_market_red --replace --chunk-months 1
  
  
  ：cn_local_industry_map_hist 作为基础表 不放在日更新job 中
  
  
D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2\daily_spot_update.bat。
改动是：日更主流程成功后、以及 auto-repair 成功后，都会追加执行一次最新交易日的大市/板块信号守卫：
daily_spot_update.bat
新增脚本：
scripts/ensure_daily_market_mainline_signal_states.py
它会找最新股票/指数共同交易日，然后检查并补齐这些表：
text



cn_local_industry_proxy_daily
cn_ga_mainline_radar_daily
cn_ga_market_pulse_daily
cn_mainline_lifecycle_daily

同时会尝试补中间依赖：
text



cn_ga_stock_role_map_daily
cn_stock_mainline_strength_daily

但这两个如果当天仍缺，不再阻塞大市/板块信号。也就是说：即使 role_map 还没补出来，大市和板块状态也会用 cn_local_industry_proxy_daily + index 兜底生成，不会再让报告停在没信号。
已验收当前最新交易日 2026-05-22：
text



cn_local_industry_proxy_daily      31 rows
cn_ga_mainline_radar_daily         31 rows
cn_ga_market_pulse_daily            1 row
cn_mainline_lifecycle_daily        31 rows

当前大市信号：
text



market_state = TREND_STRONG
market_phase = TREND_STRONG
risk_flag = LOW
market_score = 65.8662
breadth_up_ratio = 0.7201

我也跑了：
powershell



python -m py_compile scripts\ensure_daily_market_mainline_signal_states.py
python scripts\ensure_daily_market_mainline_signal_states.py

都通过。完整 daily_spot_update.bat 没有重新全量跑，避免触发整套抓取和派生链路长时间运行；这次验证的是新增的补洞模块本身。  



===
V7 
@echo off
setlocal
cd /d "%~dp0"

python scripts\ensure_daily_market_mainline_signals.py %*
if errorlevel 1 exit /b %errorlevel%

python scripts\run_p61_p64_daily_human_report.py --skip-live-observation
if errorlevel 1 exit /b %errorlevel%
