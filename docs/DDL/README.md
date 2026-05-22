# DDL Apply Guide (cn_market)

This folder stores MySQL DDL used by the rotation pipeline.

## Scope

- Database: `cn_market`
- Runtime-critical chain:
  - `SP_ROTATION_DAILY_REFRESH`
  - `sp_refresh_sector_eod_hist`
  - `SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE`
  - `SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE`
  - `cn_sector_rotation_transition_v`

## Recommended Apply Order (Rotation Core)

Apply these first for daily rotation runs:

1. `cn_market.cn_stock_universe_status_t.sql`
2. `cn_market.cn_stock_daily_price_active_v.sql`
3. `cn_market.cn_stock_active_universe_v.sql`
4. `cn_market.cn_stock_non_active_universe_v.sql`
5. `cn_market.sp_refresh_stock_universe_status.sql`
6. `cn_market.sp_refresh_sector_eod_hist.sql`
7. `cn_market.cn_sector_rotation_transition_v.sql`
8. `cn_market.sp_build_sector_rotation_ranked_by_date.sql`
9. `cn_market.sp_build_sector_rotation_signal_by_date.sql`
10. `cn_market.sp_backfill_rot_bt_from_price.sql`
11. `cn_market.sp_repair_rot_bt_nav.sql`
12. `cn_market.sp_refresh_rotation_snap_all.sql`
13. `cn_market.sp_rotation_daily_refresh.sql`

Optional wrappers:

14. `cn_market.sp_build_sector_rotation_ranked_latest.sql`
15. `cn_market.sp_build_sector_rotation_signal_latest.sql`

Optional analytics views:

16. `cn_market.cn_stock_daily_basic.sql`
17. `cn_market.cn_stock_leader_score_v1.sql`
18. `cn_market.cn_stock_leader_score_v2.sql`
19. `cn_market.cn_stock_monthly_basic.sql`
20. `cn_market.cn_stock_income.sql`
21. `cn_market.cn_stock_balancesheet.sql`
22. `cn_market.cn_stock_fina_indicator.sql`
23. `cn_market.cn_stock_cashflow.sql`
24. `cn_market.cn_fundamental_quality_param_t.sql`
25. `cn_market.cn_stock_fundamental_quality_v1.sql`
26. `cn_market.cn_stock_fundamental_quality_hist_v1.sql`
27. `cn_market.cn_stock_fundamental_quality_snap.sql`
28. `cn_market.cn_stock_financial_event_bridge_v1.sql`

Note:

- Rotation procedures that compare `run_id`/`baseline_key` explicitly declare `CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci` to match rotation table columns and avoid MySQL 8 `utf8mb4_unicode_ci` vs `utf8mb4_0900_ai_ci` comparison errors. If you hit error 1267, re-apply steps 10-13 and the validation SPs below.
- `cn_stock_leader_score_v1` and `cn_stock_leader_score_v2` are views
- on the current local DB check (`2026-05-05`), `cn_stock_leader_score_v2` is still a `VIEW`, not a table

Validation SPs:

25. `cn_market.sp_validate_sector_rot_run.sql`
26. `cn_market.sp_validate_against_baseline.sql`

Board agg views (independent):

21. `cn_market.cn_board_industry_eod_agg_v.sql`
22. `cn_market.cn_board_concept_eod_agg_v.sql`

## Copy-Paste MySQL Commands

Run from project root in MySQL client:

```sql
USE cn_market;
SOURCE docs/DDL/cn_market.cn_stock_universe_status_t.sql;
SOURCE docs/DDL/cn_market.cn_stock_daily_price_active_v.sql;
SOURCE docs/DDL/cn_market.cn_stock_active_universe_v.sql;
SOURCE docs/DDL/cn_market.cn_stock_non_active_universe_v.sql;
SOURCE docs/DDL/cn_market.sp_refresh_stock_universe_status.sql;
SOURCE docs/DDL/cn_market.sp_refresh_sector_eod_hist.sql;
SOURCE docs/DDL/cn_market.cn_sector_rotation_transition_v.sql;
SOURCE docs/DDL/cn_market.sp_build_sector_rotation_ranked_by_date.sql;
SOURCE docs/DDL/cn_market.sp_build_sector_rotation_signal_by_date.sql;
SOURCE docs/DDL/cn_market.sp_backfill_rot_bt_from_price.sql;
SOURCE docs/DDL/cn_market.sp_repair_rot_bt_nav.sql;
SOURCE docs/DDL/cn_market.sp_refresh_rotation_snap_all.sql;
SOURCE docs/DDL/cn_market.sp_rotation_daily_refresh.sql;
SOURCE docs/DDL/cn_market.cn_stock_daily_basic.sql;
SOURCE docs/DDL/cn_market.cn_stock_leader_score_v1.sql;
SOURCE docs/DDL/cn_market.cn_stock_leader_score_v2.sql;
SOURCE docs/DDL/cn_market.cn_stock_monthly_basic.sql;
SOURCE docs/DDL/cn_market.cn_stock_income.sql;
SOURCE docs/DDL/cn_market.cn_stock_balancesheet.sql;
SOURCE docs/DDL/cn_market.cn_stock_fina_indicator.sql;
SOURCE docs/DDL/cn_market.cn_stock_cashflow.sql;
SOURCE docs/DDL/cn_market.cn_fundamental_quality_param_t.sql;
SOURCE docs/DDL/cn_market.cn_stock_fundamental_quality_v1.sql;
SOURCE docs/DDL/cn_market.cn_stock_fundamental_quality_hist_v1.sql;
SOURCE docs/DDL/cn_market.cn_stock_fundamental_quality_snap.sql;
SOURCE docs/DDL/cn_market.cn_stock_financial_event_bridge_v1.sql;
```

## Tushare Load

Load market-cap and daily-basic fields from Tushare:

```powershell
python -m app.tools.sync_cn_stock_daily_basic_from_tushare --provider auto --start 2026-01-09 --end 2026-02-27 --calendar-source board-map
```

Near-to-far historical backfill:

```powershell
python -m app.tools.sync_cn_stock_daily_basic_from_tushare --provider tushare --calendar-source price --date-order desc --batch-size 20 --start 2000-01-04 --end 2026-04-29
```

This loads:

- `cn_stock_daily_basic`
- `total_mv`
- `circ_mv`
- share fields and valuation fields

Provider behavior:

- `--provider tushare`
  - Tushare only
- `--provider akshare`
  - free fallback snapshot only
- `--provider auto`
  - Tushare first, fallback to free `akshare`

Then rebuilds:

- `cn_stock_leader_score_v1`
- `cn_stock_leader_score_v2`

These are view rebuilds, not table loads.

## Weekly Runner Task

The runner now includes a stock-basic refresh task for daily use:

```powershell
python -m app.cli --tasks stock_basic --asof latest
```

Behavior:

- intended to run daily after price data is ready
- refreshes a recent rolling window by default
- can also be included in `--tasks all`

Env controls:

- `STOCK_BASIC_ENABLED`
- `STOCK_BASIC_FORCE`
- `STOCK_BASIC_PROVIDER`
- `STOCK_BASIC_CALENDAR_SOURCE`
- `STOCK_BASIC_LOOKBACK_DAYS`
- `STOCK_BASIC_DATE_ORDER`
- `STOCK_BASIC_BATCH_SIZE`
- `STOCK_BASIC_SOURCE_LABEL`
- `STOCK_BASIC_AKSHARE_WORKERS`
- `STOCK_BASIC_AKSHARE_TIMEOUT`

Compatibility:

- legacy `STOCK_BASIC_WEEKLY_*` env vars are still accepted

## Data Refresh Schedule

| 频率 | batch file | runner tasks | 数据表 |
|------|-----------|-------------|-------|
| 日 | `daily.bat` | `stock,rotation` | `cn_stock_daily_price`, rotation tables |
| 日 | `daily.bat` | `stock_basic` | `cn_stock_daily_basic` → PE_TTM / PB / PS |
| 日 | `daily.bat` | `sw_industry` | `cn_sw_industry_daily` → industry_return / industry_strength |
| 日 | `daily.bat` | `index,event` | `cn_index_daily_price`, event tables |
| 周 | `weekly.bat` | `board` | `cn_board_member_map_d` (板块成分) |
| 月 | `monthly.bat` | `stock_fundamental` | `cn_stock_monthly_basic`, `cn_stock_income`, `cn_stock_balancesheet`, `cn_stock_fina_indicator`, `cn_stock_cashflow` |

## Monthly Fundamental Task

Run monthly fundamentals (historical backfill from 2008):

```powershell
set STOCK_FUNDAMENTAL_MONTHLY_FORCE=1
set STOCK_FUNDAMENTAL_MONTHLY_HISTORY_START=20080101
python runner.py --tasks stock_fundamental --asof 20260407
```

What it loads:

- `cn_stock_monthly_basic` — month-end PE_TTM / PB / PS snapshot
- `cn_stock_income` — quarterly income statement
- `cn_stock_balancesheet` — quarterly balance sheet
- `cn_stock_fina_indicator` — ROE / `netprofit_yoy` / `or_yoy` / `ocf_to_or`
- `cn_stock_cashflow` — `n_cashflow_act` (经营活动现金流净额), used for `ocf_to_np`
- refreshes:
  - `cn_stock_fundamental_quality_v1`
  - `cn_stock_fundamental_quality_hist_v1`
  - `cn_stock_fundamental_quality_snap` (materialized; adds `roe`, `netprofit_yoy`, `ocf_to_np`)

Fetch modes for quarterly financial tables (income / balancesheet / fina_indicator / cashflow):

- **Default (incremental)** — disclosure-date driven: reads `cn_event_disclosure_date` to know which symbols disclosed on each date, then calls `api(ts_code=..., start_date=..., end_date=...)`. Correct for daily catch-up where the disclosure table is populated.
- **`--by-symbol` (historical backfill)** — iterates over all symbols in `cn_stock_daily_price` for the requested period (includes delisted stocks), calls `api(ts_code=..., start_date=..., end_date=...)` per symbol. Use when `cn_event_disclosure_date` has no rows for the target period.

Thresholds driven by:

- `cn_fundamental_quality_param_t`

Standalone historical backfill flag:

- `--by-symbol` — iterate over every symbol in `cn_stock_daily_price` (includes delisted stocks from the period) and call `api(ts_code=..., start_date=..., end_date=...)` per symbol. This is the correct primary query dimension for quarterly financial APIs. Use for any historical period where `cn_event_disclosure_date` is empty. Applies to all four tables: income, balancesheet, fina\_indicator, cashflow.

Env controls:

- `STOCK_FUNDAMENTAL_MONTHLY_CASHFLOW_SOURCE_LABEL` (default: `tushare_cashflow`)
- `STOCK_FUNDAMENTAL_MONTHLY_BY_SYMBOL` (default: `0`) — set to `1` to use symbol-based fetch (historical backfill mode) in runner task

Operational rule update:

- For quarterly financial APIs, treat `ts_code` as the primary fetch entrance.
- Incremental mode remains disclosure-driven for scheduling, but the actual fetch shape is now `symbol-first`.
- The loader groups disclosure dates by `symbol`, requests one bounded date window per `symbol`, then filters locally by disclosure dates.
- Historical backfill with `--by-symbol` keeps the same `symbol-first` shape and simply widens the date window.
- Long-running loaders print launch banner plus rolling progress / ETA in console.

## SW Industry Daily Task

Syncs Shenwan L1 industry daily price and valuation. Run daily after stock price data.

```powershell
python runner.py --flag tu --tasks sw_industry --asof latest
```

What it loads:

- `cn_sw_industry_daily` — `pct_change`, `close`, `pe`, `pb`, `float_mv`

Views rebuilt on-the-fly (no extra step needed):

- `cn_sw_industry_daily_latest_v`
- `cn_sw_industry_strength_latest_v` — exposes `industry_return` (= `rs_20d`) and `industry_strength` (= `trend_state`)

Env controls:

- `SW_INDUSTRY_DAILY_ENABLED` (default: `1`)
- `SW_INDUSTRY_DAILY_LOOKBACK_DAYS` (default: `3`)
- `SW_INDUSTRY_DAILY_FORCE` (default: `0`)
- `SW_INDUSTRY_DAILY_SRC` (default: `SW2021`)
- `SW_INDUSTRY_DAILY_MASTER_SOURCE` (default: `TUSHARE_SW2021_L1`)
- `SW_INDUSTRY_DAILY_SLEEP` (default: `7.0` s between codes — sw_daily limit is 10/min)
- `SW_INDUSTRY_DAILY_SOURCE_LABEL`

Historical backfill:

```powershell
set SW_INDUSTRY_DAILY_FORCE=1
set SW_INDUSTRY_DAILY_LOOKBACK_DAYS=9999
python runner.py --flag tu --tasks sw_industry --asof 20260430
```

## Historical Backfill (2010-01-01 → 2026-04-30)

Run in order. Step 1 and Step 2 are independent and can run in parallel terminals.

### Step 1 — `cn_stock_daily_basic` (PE_TTM / PB / PS)

```powershell
python -m app.tools.sync_cn_stock_daily_basic_from_tushare `
  --provider tushare `
  --calendar-source price `
  --date-order desc `
  --batch-size 20 `
  --start 2010-01-01 `
  --end 2026-04-30
```

### Step 2 — `cn_sw_industry_daily` (industry\_return / industry\_strength source)

```powershell
python -m app.tools.sync_cn_sw_industry_daily_from_tushare `
  --start 2010-01-01 `
  --end 2026-04-30 `
  --full
```

### Step 3 — Quarterly financials + cashflow (all four tables, symbol-based)

`--by-symbol` iterates over all symbols in `cn_stock_daily_price` for the period — this includes
stocks that have since been delisted, which a current-universe query would miss.
It calls `api(ts_code=..., start_date=..., end_date=...)` per symbol, which is the primary query
dimension for income / balancesheet / fina_indicator / cashflow APIs.

```powershell
python -m app.tools.sync_cn_stock_fundamental_monthly `
  --provider tushare `
  --start 2010-01-01 `
  --end 2026-04-30 `
  --by-symbol
```

Or via runner (uses `STOCK_FUNDAMENTAL_MONTHLY_BY_SYMBOL` env var):

```powershell
$env:STOCK_FUNDAMENTAL_MONTHLY_FORCE = "1"
$env:STOCK_FUNDAMENTAL_MONTHLY_HISTORY_START = "20100101"
$env:STOCK_FUNDAMENTAL_MONTHLY_BY_SYMBOL = "1"
python runner.py --flag tu --tasks stock_fundamental --asof 20260430
```

### Step 4 — Rebuild quality snapshot

```powershell
$env:STOCK_FUNDAMENTAL_MONTHLY_FORCE = "1"
python runner.py --flag tu --tasks stock_fundamental --asof 20260430
```

## Quick Verification

```sql
SELECT ROUTINE_NAME
FROM information_schema.routines
WHERE routine_schema='cn_market'
  AND routine_name IN (
    'SP_ROTATION_DAILY_REFRESH',
    'SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE',
    'SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE',
    'sp_refresh_sector_eod_hist'
  );

SELECT TABLE_NAME
FROM information_schema.views
WHERE table_schema='cn_market'
  AND table_name IN (
    'cn_sector_rotation_transition_v',
    'cn_stock_leader_score_v1',
    'cn_stock_leader_score_v2',
    'cn_stock_fundamental_quality_v1',
    'cn_stock_fundamental_quality_hist_v1'
  );

SELECT COUNT(*)
FROM cn_stock_fundamental_quality_snap;
```

## Notes

- If you hit MySQL 1146 on `cn_sector_rotation_transition_v`, apply
  `cn_market.cn_sector_rotation_transition_v.sql` and rerun rotation.
- `Dump_c_market_20260305.sql` is a full dump artifact, not part of the
  incremental apply order above.
- Leader-score queries should use the latest available industry-mapping date.
