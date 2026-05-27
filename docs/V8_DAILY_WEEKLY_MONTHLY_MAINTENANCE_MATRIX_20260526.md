# V8 Daily / Weekly / Monthly Maintenance Matrix

Date: 2026-05-26
Scope: AsharesScraper data-maintenance boundary for GrowthAlpha V8 / P18+ operational use.

## Goal

Keep daily maintenance focused on market freshness and operational state. Move slower financial-quality / alpha refreshes to weekly or monthly so `daily_spot_update.bat` does not become a heavy all-in-one backfill job.

## Ownership boundary

| Layer | Owner | Responsibility | Not responsible for |
|---|---|---|---|
| AsharesScraper | Data supply / reusable derived tables | Raw A-share market data, index data, daily_basic, leader score materialization, role map, mainline/radar/pulse/lifecycle, financial-quality snapshots, unified alpha refresh | GrowthAlpha P64/P65/P70 operational reports, live-observation SQLite report rendering |
| GrowthAlpha | Strategy / operation layer | Current holding interpretation, P64/P65/P70 reports, daily operational report, live-observation SQLite writes, research audit outputs | Tushare pulling, raw quote repair, generic reusable V8 market data maintenance |

## Daily maintenance

Default script: `daily_spot_update.bat`

Daily is a **light operational refresh**.

| Task | Frequency | Reason |
|---|---:|---|
| `v8_daily_market_raw` | Daily | Fresh stock/index prices are required for all downstream state. |
| `v8_daily_reference` | Daily, light | SW daily disabled by default; event and legacy rotation skipped by default. |
| `v8_stock_basic` | Daily | Updates `cn_stock_daily_basic` and materializes leader score for latest dates. |
| `v8_daily_audit` | Daily | Verifies stock/index coverage. |
| `v8_daily_derived_foundation` | Daily, partial | Runs capital flow and role map; skips fundamental/quality. |
| `v8_daily_derived_mainline` | Daily | Updates mainline strength, radar, market pulse, local industry proxy, lifecycle. |
| `v8_daily_derived_alpha` | Not daily by default | Moved to weekly/monthly because it depends on financial-quality chain. |

Daily default skip flags:

```bat
set V8_SKIP_STOCK_FUNDAMENTAL_DAILY=1
set V8_SKIP_STOCK_QUALITY_SCORE=1
set V8_SKIP_UNIFIED_ALPHA=1
```

Daily retained flags:

```bat
set V8_SKIP_INDUSTRY_CAPITAL_FLOW=0
set V8_SKIP_GA_STOCK_ROLE_MAP=0
set V8_SKIP_MAINLINE_STRENGTH=0
set V8_SKIP_MAINLINE_RADAR=0
set V8_SKIP_MARKET_PULSE=0
set V8_SKIP_LOCAL_INDUSTRY_PROXY=0
set V8_SKIP_MAINLINE_LIFECYCLE=0
```

## Weekly maintenance

Default script: `weekly.bat`

Weekly now owns quality/alpha catch-up.

| Task | Frequency | Reason |
|---|---:|---|
| `v8_stock_basic` | Weekly | Backstop daily_basic / leader score gaps. |
| `event_periodic` | Weekly | Periodic event data is not required every day. |
| `v8_weekly_audit_market` | Weekly | Wider market coverage audit. |
| `v8_monthly_derived` | Weekly | Refreshes financial snapshot and stock quality score for recent window. |
| `v8_daily_derived_alpha` | Weekly | Refreshes unified alpha after quality data is updated. |
| `v8_weekly_finalize` | Weekly | Latest leader snapshot / optional crosswalk finalize. |

## Monthly maintenance

Default script: `monthly.bat`

Monthly owns slower fundamental and map maintenance.

| Task | Frequency | Reason |
|---|---:|---|
| `v8_monthly_refresh` | Monthly | Fundamental monthly pull and periodic events. |
| `v8_monthly_audit` | Monthly | Long-window coverage audit. |
| `v8_monthly_derived` | Monthly | Financial snapshot, stock quality score, and monthly alpha refresh. |
| `build_local_industry_map_hist.py` L1/L2/L3 | Monthly incremental by default | Industry map history should not full-backfill every monthly run. |

Monthly map-history defaults:

```bat
set V8_MONTHLY_MAP_LOOKBACK_DAYS=365
set V8_MONTHLY_FULL_MAP_HIST=0
```

For manual full map backfill only:

```bat
set V8_MONTHLY_FULL_MAP_HIST=1
monthly.bat --refresh
```

## Emergency full daily repair

Daily auto-repair remains light by default. If the operator explicitly needs a full daily repair including financial-quality-alpha chain:

```bat
set V8_DAILY_REPAIR_FULL_DERIVED=1
daily_spot_update.bat
```

## P0 fixes included

1. `daily_spot_update.bat` was simplified to avoid nested `if (...)` parser failures caused by comments/commands with parentheses inside blocks.
2. `monthly.bat` now returns the real `%ERRORLEVEL%` on failure instead of always returning 0.
3. `app/tasks/v8_dataset_ops_task.py` now honors `V8_SKIP_UNIFIED_ALPHA=1` inside `_run_derived_alpha_chain()`.

## Operational acceptance

Daily run is acceptable when:

- stock/index audit reports no fatal gaps;
- `cn_stock_daily_price`, `cn_stock_daily_basic`, `cn_stock_leader_score_daily` reach latest trade date;
- `cn_ga_stock_role_map_daily`, `cn_stock_mainline_strength_daily`, `cn_ga_mainline_radar_daily`, `cn_ga_market_pulse_daily`, `cn_local_industry_proxy_daily`, and `cn_mainline_lifecycle_daily` update for latest buildable date;
- no financial-quality or unified-alpha failure blocks daily completion.

Weekly/monthly run is acceptable when:

- `cn_stock_fundamental_daily`, `cn_stock_quality_score_daily`, and `cn_unified_alpha_score_daily` refresh for the requested recent window;
- event periodic and monthly fundamental tasks complete;
- map-history incremental refresh completes or fails with a non-zero exit code.
