# V8 Future-System Update Runbook

Date: 2026-05-21 (updated 2026-05-21)

## Purpose

This runbook defines the current operational update contract for the future-system
path in `AsharesScraperV2`.

The goal is to keep refreshing data still used by the V8 / GrowthAlpha future
path, while stopping default refresh of legacy compatibility tables that are no
longer required in routine daily/weekly/monthly production updates.

## Primary Rule

Default production update chains should keep refreshing:

- `cn_stock_daily_price`
- `cn_index_daily_price`
- `cn_stock_daily_basic`
- `cn_sw_industry_daily`
- `cn_stock_monthly_basic`
- `cn_stock_fina_indicator`
- `cn_stock_income`
- `cn_stock_balancesheet`
- `cn_stock_cashflow`
- `cn_event_*`
- `cn_stock_fundamental_daily`
- `cn_stock_quality_score_daily`
- `cn_industry_capital_flow_daily`
- `cn_ga_stock_role_map_daily`
- `cn_stock_mainline_strength_daily`
- `cn_ga_mainline_radar_daily`
- `cn_ga_market_pulse_daily`
- **`cn_local_industry_proxy_daily`** ← added 2026-05-21 (daily derived, required by lifecycle)
- `cn_mainline_lifecycle_daily`
- `cn_unified_alpha_score_daily`
- `cn_stock_leader_sw_l1_latest_snap`

Default production update chains should not refresh legacy compatibility assets:

- `cn_board_concept_member_hist`
- `cn_board_industry_member_hist`
- `cn_board_member_map_d`
- legacy rotation snapshot tables driven by `SectorRotationSnapshotTask`
- `cn_v7_v8_industry_crosswalk_latest`

Notes:

- `cn_v7_v8_industry_crosswalk` and `cn_v7_v8_industry_crosswalk_latest` are still
  valid compatibility artifacts, but they are no longer refreshed by default in
  routine future-system ops.
- `cn_board_*` tables are kept for legacy/research/recovery workflows only.
- For current V8 database semantics, also see:
  - [V8_LOCAL_INDUSTRY_SEMANTICS_CONTRACT_20260525.md](./V8_LOCAL_INDUSTRY_SEMANTICS_CONTRACT_20260525.md)
  - [V8_CROSSWALK_SCOPE_GUIDE_20260525.md](./V8_CROSSWALK_SCOPE_GUIDE_20260525.md)

### Static Base Tables (refreshed monthly, not daily)

These tables are **static mapping tables** that do not change on a daily basis.
They are refreshed monthly via `monthly.bat` rather than in the daily pipeline:

- `cn_local_industry_map_hist` — stock ↔ SW industry membership with in_date/out_date.
  Sourced from Tushare `index_member_all`. Only updated when industry classifications
  change (e.g., new stocks added to an industry).

Current V8 semantic interpretation:

- `cn_local_industry_map_hist.industry_level = 'L3'` = `LOCAL_FINE`, the
  fine-grained V8 production membership layer
- `cn_local_industry_proxy_daily.industry_level = 'L1'` = legacy physical label
  for the same `LOCAL_FINE` production layer
- `SW_L1` keeps its literal meaning and refers to the official Shenwan level-1
  comparison layer
- `scripts/build_local_industry_map_hist.py` defaults to `--level L3`
  for the V8 production path

## Current Batch Contract

### Daily

Entrypoint:

```bat
daily_spot_update.bat
```

**职责**：高频原始数据、Rotation/Mainline/Alpha

For the current V8 production contract:

- `L3` is the required `LOCAL_FINE` membership layer
- `L1/L2` are optional compatibility/reference refreshes

Current default behavior:

- refresh market raw tables（`StockLoaderTask`、`IndexLoaderTask`）
- refresh `sw_industry`（`SwIndustryDailyTask`）
- skip daily event loader by default
- skip rotation snapshot by default
- run **daily derived chain**（`_run_daily_derived_chain`）：
  - **Mainline Chain**：mainline strength、radar、market pulse、local industry proxy、lifecycle
  - **Industry Capital Flow**：`build_industry_capital_flow_daily.py`
  - **Stock Role Map**：`build_ga_stock_role_map_daily.py`
  - **Alpha Chain**：`build_unified_alpha_score_daily.py`
- keep `crosswalk_latest` disabled

Default legacy exclusions:

- `V8_SKIP_ROTATION_SNAPSHOT=1`
- `V8_ENABLE_CROSSWALK_LATEST=0`
- `V8_DAILY_INCLUDE_CROSSWALK_LATEST=0`

Reason:

- `SectorRotationSnapshotTask` still depends on `cn_board_member_map_d` and legacy
  rotation tables, so it is not part of the future-system default daily refresh.

### Pipeline 职责划分

| Pipeline | 重点 | 原始数据 | 衍生链 |
|----------|------|---------|--------|
| **Daily** | 高频原始数据、Rotation/Mainline/Alpha | Stock/Index/SW Industry | Mainline Chain + Industry Capital Flow + Stock Role Map + Alpha Chain |
| **Weekly** | 低频 Reference/Event、Board/Event/Mapping | Stock Basic / Event Periodic / Board Membership | 不调衍生链，仅做 snap + crosswalk |
| **Monthly** | 财务报表与质量因子 | Stock Fundamental Monthly / Event Periodic | Fundamental Daily + Quality Score（Alpha Chain 可选，默认关闭） |

> ⚠️ **核心原则**：Monthly pipeline **不补任何日频数据**。日频数据（行情、轮动、主线、Alpha）的回补统一由 `daily.bat` 负责。Monthly 只负责低频财务数据刷新和基于财务数据的质量评分。

#### Mainline Chain Execution Order

The `_run_derived_mainline_chain` now executes in this order:

1. `build_cn_stock_mainline_strength_daily.py` — stock-level mainline strength
2. `build_ga_mainline_radar_daily.py` — mainline radar (role-based scoring)
3. `build_ga_market_pulse_daily.py` — market pulse indicators
4. **`build_local_industry_proxy_daily.py`** — industry proxy (member_count, ret_eqw, etc.)
5. `build_mainline_lifecycle_daily.py` — lifecycle state machine (depends on proxy data)

Step 4 is controlled by `V8_SKIP_LOCAL_INDUSTRY_PROXY` (default: 0 = run).

### Weekly

Entrypoint:

```bat
weekly.bat
```

Current default task list:

1. `v8_stock_basic`
2. `event_periodic`
3. `v8_weekly_audit_market`
4. `v8_weekly_finalize`

Current default behavior:

- refresh `cn_stock_daily_basic`
- rebuild leader-score materialization
- refresh periodic event tables
- run stock/index market audit only
- refresh latest leader SW L1 snapshot
- do not refresh board membership tables
- do not run board-reference audit
- do not refresh `crosswalk_latest`

Default legacy exclusions:

- `V8_WEEKLY_BUILD_CROSSWALK_LATEST=0`
- `V8_ENABLE_CROSSWALK_LATEST=0`
- no call to `v8_weekly_refresh`
- no call to `v8_weekly_audit`

Reason:

- `v8_weekly_refresh` bundles `BoardMembershipRefreshTask`, which writes
  `cn_board_*` legacy tables.
- `v8_weekly_audit` asserts `cn_board_concept_member_hist`,
  `cn_board_industry_member_hist`, and `cn_board_member_map_d`, so it is not
  suitable once those tables are no longer part of the routine future-system
  refresh contract.

#### Weekly 参数说明

`weekly.bat` 透传所有参数给 `runner.py`，以下两个参数容易混淆：

| 参数 | 作用范围 | 格式 | 用途 |
|------|---------|------|------|
| `--start-date` / `--end-date` | 所有 task（`v8_weekly_*`、`v8_daily_*`、`v8_backfill` 等） | `YYYYMMDD` 或 `YYYY-MM-DD` | 显式指定日期窗口，覆盖 `--days` 的自动计算 |
| `--history-start` / `--history-end` | **仅限** `his_stocks` task | `YYYYMMDD` 或 `YYYY-MM`（支持月份简写） | 指定 `HisStocksLoaderTask` 的历史数据拉取起始点 |

**关键区别**：

- `--start-date` 写入 `cfg.start_date` / `cfg.end_date`，被所有 task 共用（[`cli.py` 第 194-200 行](../../app/cli.py:194)）
- `--history-start` 写入 `cfg.his_start_date` / `cfg.his_end_date`，**仅当 `his_stocks` 在 task 列表中时生效**（[`cli.py` 第 209-225 行](../../app/cli.py:209)），同时也会覆盖 `cfg.start_date` / `cfg.end_date`

**常见场景**：

- **补跑 V8 周度数据**（如从 2026-01-01 补到今天）：使用 `--start-date`
  ```bat
  weekly.bat --start-date 20260101 --end-date latest
  ```
  这会覆盖 `weekly.bat` 默认的 `--asof latest --days 7`（最近 7 天），改为从 2026-01-01 跑到最新交易日。

- **补历史行情数据**（原始日线行情回填）：使用 `--history-start`，且 task 必须为 `his_stocks`
  ```bat
  python runner.py --tasks his_stocks --history-start 20260101 --history-end latest
  ```

> ⚠️ 注意：`--history-start` 对 `v8_weekly` 等非 `his_stocks` task **完全无效**。如果你跑 `weekly.bat` 时传了 `--history-start`，该参数会被忽略。

### Monthly

Entrypoint:

```bat
monthly.bat
```

Monthly Pipeline 总体结构：

```
v8_monthly_refresh
        ↓
v8_monthly_audit
        ↓
v8_monthly_derived
        ↓
build_local_industry_map_hist (L1/L2/L3)
```

#### 1. `v8_monthly_refresh` — 月度基础财务数据刷新

主要负责更新低频财务和估值数据：

- **财务报表**：`cn_stock_income`、`cn_stock_balancesheet`、`cn_stock_cashflow`、`cn_stock_fina_indicator`
- **月度估值**：`cn_stock_monthly_basic`
- **财务质量参数**：`cn_fundamental_quality_param_t`
- **Periodic 事件**：`EventLoaderPeriodic`（分红、预告等）

#### 2. `v8_monthly_audit` — 财务数据完整性审计

检查内容：
- 最近报告期覆盖率
- 股票覆盖率
- 空值比例 / 异常值检查
- 行数阈值

输出：
- `cn_ga_data_readiness_daily`
- `monthly_financial_audit.csv`

#### 3. `v8_monthly_derived` — 财务报表与质量因子衍生链

`V8MonthlyDerivedTask` 通过 `_run_monthly_derived_chain()` 执行，**仅刷新财务报表和质量因子**，不涉及行情/轮动/主线等日频数据。

| 脚本 | 构建表 | 说明 |
|------|--------|------|
| `build_stock_fundamental_daily.py` | `cn_stock_fundamental_daily` | 每日基本面快照（ROE/ROIC/毛利率/净利增长/现金流/杠杆） |
| `build_stock_quality_score_daily.py` | `cn_stock_quality_score_daily` | 质量评分 |
| `build_unified_alpha_score_daily.py` | `cn_unified_alpha_score_daily` | 统一 Alpha 评分（**可选**，`V8_MONTHLY_INCLUDE_ALPHA=1` 时启用，默认关闭） |
| `build_v7_v8_crosswalk_latest.py` | `cn_v7_v8_industry_crosswalk` | V7↔V8 行业映射（`V8_MONTHLY_INCLUDE_CROSSWALK_LATEST=1` 时启用） |
| `validate_unified_alpha_leader_recall.py` | — | Leader 回测验证（`V8_ENABLE_LEADER_RECALL_VALIDATION=1` 时启用） |

> ⚠️ **核心原则**：Monthly pipeline **不补任何日频数据**。日频数据（行情、轮动、主线、Alpha）的回补统一由 `daily.bat` 负责。
>
> - `_run_latest_snap()`（`cn_stock_leader_sw_l1_latest_snap`）由 Weekly pipeline 的 `v8_weekly_finalize` 负责
> - `_run_derived_alpha_chain()`（`build_unified_alpha_score_daily.py`）默认由 Daily pipeline 负责，Monthly 仅在 `V8_MONTHLY_INCLUDE_ALPHA=1` 时触发
> - Mainline chain（主线强度/雷达/市场脉搏/生命周期）、industry_capital_flow、ga_stock_role_map 等日频衍生**完全不在 Monthly 范围内**

#### 4. `build_local_industry_map_hist` — 静态行业映射表

在 v8 任务链完成后运行，刷新 `cn_local_industry_map_hist`（L1/L2/L3）。

Current default behavior:

- refresh monthly/fundamental source tables
- refresh periodic event tables
- run monthly market audit
- run derived refresh chain
- keep `crosswalk_latest` disabled
- **build `cn_local_industry_map_hist`** (`L3` required for V8 LOCAL_FINE; `L1/L2` optional, after v8 tasks)

Default legacy exclusions:

- `V8_MONTHLY_INCLUDE_CROSSWALK_LATEST=0`
- `V8_ENABLE_CROSSWALK_LATEST=0`

#### Monthly Static Table Refresh

After the v8 task chain completes, `monthly.bat` runs:

```bat
python scripts/build_local_industry_map_hist.py --start 1990-01-01 --end 2026-12-31 --level L3 --resume
```

These refresh the stock ↔ SW industry membership mapping from Tushare
`index_member_all`. `--resume` skips already-completed industry chunks.

V8 semantic note:

- the required production refresh above is `--level L3`
- this corresponds to the `LOCAL_FINE` membership layer used by the V8 mainline
  production path
- `--level L1` and `--level L2` are optional compatibility/reference refreshes,
  not required for the core V8 daily proxy chain

Optional compatibility/reference refreshes:

```bat
python scripts/build_local_industry_map_hist.py --start 1990-01-01 --end 2026-12-31 --level L1 --resume
python scripts/build_local_industry_map_hist.py --start 1990-01-01 --end 2026-12-31 --level L2 --resume
```

## Data Dependency & Refresh Strategy

### Table Classification

| Table | Type | Refresh | Pipeline | Audit Strategy |
|---|---|---|---|---|
| `cn_local_industry_map_hist` | Static mapping | Monthly | `monthly.bat` | Not audited in daily pipeline (no trade_date column) |
| `cn_local_industry_proxy_daily` | Daily derived | Daily | `daily_spot_update.bat` (mainline chain) | Audits `cn_stock_daily_price` and `cn_stock_daily_basic` only |
| `cn_mainline_lifecycle_daily` | Daily derived | Daily | `daily_spot_update.bat` (mainline chain) | Audits upstream radar/strength/flow/pulse/proxy tables |

### Dependency Flow

```
cn_stock_daily_price ─┐
                      ├──> build_local_industry_proxy_daily ──┐
cn_local_industry_map_hist ┘                                  │
                                                              ├──> build_mainline_lifecycle_daily
cn_ga_mainline_radar_daily ───────────────────────────────────┘
cn_stock_mainline_strength_daily ─────────────────────────────┘
cn_industry_capital_flow_daily ───────────────────────────────┘
cn_ga_market_pulse_daily ─────────────────────────────────────┘
```

## Legacy Recovery / Research Workflows

These are still allowed, but should be run explicitly rather than through the
default future-system batch files:

- `python runner.py --tasks board ...`
- `python runner.py --tasks v8_board_refresh ...`
- `python runner.py --tasks rotation ...`
- `python runner.py --tasks v8_rotation ...`
- `python runner.py --tasks v8_weekly_refresh ...`
- `python runner.py --tasks v8_weekly_audit ...`
- `python scripts/build_v7_v8_crosswalk_latest.py --replace`

Use those only when you intentionally need:

- board/concept history maintenance
- legacy rotation snapshot maintenance
- V7/V8 compatibility latest crosswalk outputs

## Recommended Operator Overrides

If you need to temporarily re-enable a legacy path for investigation:

```bat
set V8_SKIP_ROTATION_SNAPSHOT=0
set V8_ENABLE_CROSSWALK_LATEST=1
set V8_WEEKLY_BUILD_CROSSWALK_LATEST=1
set V8_MONTHLY_INCLUDE_CROSSWALK_LATEST=1
```

If you need to skip the industry proxy builder in the daily pipeline:

```bat
set V8_SKIP_LOCAL_INDUSTRY_PROXY=1
daily_spot_update.bat
```

Signal-guard note:

- `daily_spot_update.bat` now also refreshes a recent `LOCAL_FINE`
  `cn_local_industry_map_hist --level L3` window by default before signal guard
- `daily_spot_update.bat` now also refreshes `cn_board_member_map_d` via
  `build_board_map_by_year.py` before signal guard, because
  `sp_materialize_leader_score` (called by `v8_stock_basic`) depends on
  `cn_board_member_map_d` having `INDUSTRY` records for the target dates.
  Without this, leader scores silently skip recent dates, cascading into empty
  `cn_ga_stock_role_map_daily` / `cn_stock_mainline_strength_daily` /
  `cn_ga_mainline_radar_daily` / `cn_ga_market_pulse_daily` for those dates.
- `daily_spot_update.bat` re-runs `v8_stock_basic` (to refresh leader scores)
  for the full `START_DATE..END_DATE` window before signal guard.
- `daily_spot_update.bat` re-runs `v8_daily_derived_foundation` with
  `V8_SKIP_STOCK_FUNDAMENTAL_DAILY=1`, `V8_SKIP_STOCK_QUALITY_SCORE=1`,
  `V8_SKIP_INDUSTRY_CAPITAL_FLOW=1` — only `build_ga_stock_role_map_daily.py`
  (role_map) runs, because it is the only foundation builder affected by board
  map refresh. This eliminates ~5 minutes of redundant computation
  (stock_fundamental_daily, stock_quality_score, industry_capital_flow were
  already built in the first pass).
- `daily_spot_update.bat` re-runs `v8_daily_derived_mainline` (to fill mainline
  tables) for the full `START_DATE..END_DATE` window before signal guard.
- `daily_spot_update.bat` ends by running
  `scripts/ensure_daily_market_mainline_signal_states.py`
- this guard uses the latest **buildable** mainline-signal date, not just the
  latest raw market date
- it is bounded by raw stock/index coverage, `cn_stock_daily_basic`, and the
  current `LOCAL_FINE` map horizon from
  `cn_local_industry_map_hist.industry_level = 'L3'`
- `_latest_signal_trade_date()` uses `MAX(COALESCE(out_date, in_date))` instead
  of `MAX(out_date)` for the LOCAL_FINE horizon, because `out_date` is the date
  a stock *leaves* an industry — stocks still in their industry have `NULL`
  out_date, which would incorrectly pull the max down.

If you need to restore legacy board refresh explicitly, do not use the default
`weekly.bat`. Run the board-specific tasks directly.

## Manual Backfill Commands

### cn_local_industry_map_hist (static mapping, monthly refresh)

```bat
python scripts/build_local_industry_map_hist.py --start 1990-01-01 --end 2026-12-31 --level L3 --resume
```

Optional compatibility/reference refreshes:

```bat
python scripts/build_local_industry_map_hist.py --start 1990-01-01 --end 2026-12-31 --level L1 --resume
python scripts/build_local_industry_map_hist.py --start 1990-01-01 --end 2026-12-31 --level L2 --resume
```

### cn_local_industry_proxy_daily (daily derived, standalone backfill)

```bat
python scripts/build_local_industry_proxy_daily.py --start 2026-05-01 --end 2026-05-21 --workers 4
```

## Related Docs

- [V8_DATASET_RUNBOOK_20260515.md](./V8_DATASET_RUNBOOK_20260515.md)
- [V8_LOCAL_INDUSTRY_SEMANTICS_CONTRACT_20260525.md](./V8_LOCAL_INDUSTRY_SEMANTICS_CONTRACT_20260525.md)
- [V8_CROSSWALK_SCOPE_GUIDE_20260525.md](./V8_CROSSWALK_SCOPE_GUIDE_20260525.md)
- [BK_TS_SW_ACTIVE_DEPENDENCY_AUDIT_20260521.md](./BK_TS_SW_ACTIVE_DEPENDENCY_AUDIT_20260521.md)
- [board_membership_refresh_playbook.md](./board_membership_refresh_playbook.md)
