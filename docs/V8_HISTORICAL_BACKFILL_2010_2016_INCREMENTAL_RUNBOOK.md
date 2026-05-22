# V8 Historical Backfill 2010-2016 Incremental Runbook

Date: 2026-05-17

## Purpose

This runbook consolidates the usable parts of:

- `docs/V8_TASK_SPLIT_RUNBOOK_20260515.md`
- `docs/V8_DATASET_RUNBOOK_20260515.md`
- `docs/DDL/README.md`
- related builder scripts under `scripts/`, `app/tools/`, and `data_pipeline/builders/`

It is intended for two bootstrap scenarios:

1. a V7-style database that needs V8-compatible historical backfill
2. a nearly empty / new MySQL database that must be built up for V8 historical use

This document is intentionally written for **incremental historical backfill** over
`2010-01-01` to `2016-12-31`.

The main rule is:

- prefer `--refresh`, `--resume`, and upsert-style rebuilds
- do **not** default to full-table destructive rebuilds
- use year-split execution for long windows

## What The Old Split Runbook Misses

`v8_backfill` covers the raw market / raw financial / derived mainline chain, but it does **not** fully cover the V8 compatibility mapping chain.

For 2010-2016 historical rebuild, you must explicitly include these extra tables:

- `cn_local_industry_master`
- `cn_local_industry_map_hist`
- `cn_local_industry_proxy_daily`
- `cn_ts_sw_industry_master`
- `cn_ts_sw_industry_member_hist`
- `cn_v7_v8_industry_crosswalk`
- `cn_v7_v8_industry_crosswalk_latest`
- `cn_stock_v8_to_v7_sw_map_latest`

Also note:

- `cn_stock_leader_sw_l1_latest_snap` is a latest snapshot table, not a full-history table
- `cn_v7_v8_industry_crosswalk_latest` and `cn_stock_v8_to_v7_sw_map_latest` are latest snapshots, not full-history tables
- `cn_stock_leader_score_v1` / `cn_stock_leader_score_v2` are views, but they still need to exist because downstream scripts read from the leader-score layer

## Scope

Historical target window used in this runbook:

- start: `2010-01-01`
- end: `2016-12-31`

Recommended chunking strategy:

- raw price / index / board / derived daily chains: by year
- financial sync: whole range or by year if needed
- local industry / crosswalk builders: whole range, with `--resume`

## Strategy

Use four phases:

1. schema / procedure bootstrap for empty DB
2. raw history and board/reference history
3. V8 compatibility mapping history
4. downstream derived history and latest snapshots

If the database already has the required stored procedures, views, and core tables, skip Phase 1.

## Phase 1: Empty-DB Bootstrap

### 1.1 Rotation core DDL and views

If the database is empty or missing rotation stored procedures/views, apply the rotation DDL bundle described in `docs/DDL/README.md`.

Run these in MySQL client from repo root:

```sql
USE cn_market_red;
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

### 1.2 GA-layer schema

For empty DB or missing GA/P2 tables:

```powershell
python scripts/apply_ga_p0_schema_migration.py ^
  --db-host 127.0.0.1 ^
  --db-port 3306 ^
  --db-user cn_opr_red ^
  --db-password YOUR_PASSWORD ^
  --db-name cn_market_red
```

### 1.3 Board membership stored procedures

`board` refresh and historical board-map rebuild depend on:

- `sp_refresh_board_member_hist`
- `sp_build_board_member_map`

Those procedures are referenced by:

- [board_membership_refresh_task.py](/d:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/app/tasks/board_membership_refresh_task.py)
- [build_board_member_map_full.py](/d:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/scripts/build_board_member_map_full.py)

If they are missing on a brand-new DB, restore them from your DBA-managed schema source before continuing.

## Phase 2: Raw History And Board / Reference History

### 2.1 Price history

Preferred unified path:

```powershell
set V8_BACKFILL_START=20100101
set V8_BACKFILL_END=20161231
set V8_BACKFILL_INCLUDE_PRICE=1
set V8_BACKFILL_INCLUDE_FUNDAMENTAL=0
set V8_BACKFILL_INCLUDE_DAILY_BASIC=0
set V8_BACKFILL_INCLUDE_SW_DAILY=0
set V8_BACKFILL_INCLUDE_BOARD_REFRESH=0
set V8_BACKFILL_INCLUDE_DERIVED=0
python runner.py --flag tu --tasks v8_backfill --asof 20161231
```

If you want strict year-split execution:

```powershell
stock_daily_price_by_year.bat 2010 2016
```

### 2.2 Index history

```powershell
python runner.py --flag tu --tasks index --start-date 2010-01-01 --end-date 2016-12-31 --refresh
```

### 2.3 Daily basic + raw financials + SW daily

```powershell
set V8_BACKFILL_START=20100101
set V8_BACKFILL_END=20161231
set V8_BACKFILL_INCLUDE_PRICE=0
set V8_BACKFILL_INCLUDE_FUNDAMENTAL=1
set V8_BACKFILL_INCLUDE_DAILY_BASIC=1
set V8_BACKFILL_INCLUDE_SW_DAILY=1
set V8_BACKFILL_INCLUDE_BOARD_REFRESH=0
set V8_BACKFILL_INCLUDE_DERIVED=0
python runner.py --flag tu --tasks v8_backfill --asof 20161231
```

Alternative direct SW daily command:

```powershell
python scripts/backfill_sw_industry_daily.py --start 2010-01-01 --end 2016-12-31 --resume
```

### 2.4 Periodic and daily events

Daily events:

```powershell
python runner.py --flag tu --tasks v8_event_daily --start-date 2010-01-01 --end-date 2016-12-31 --refresh
```

Periodic events:

```powershell
event_backfill.bat --start-date 2010-01-01 --end-date 2016-12-31
```

### 2.5 Board membership history and board map

First refresh the board history chain through runner:

```powershell
weekly_by_year.bat 2010 2016 --refresh
```

Then rebuild `cn_board_member_map_d` serially to avoid lock-contention:

```powershell
python scripts/build_board_member_map_full.py ^
  --start 2010-01-01 ^
  --end 2016-12-31 ^
  --resume ^
  --db-host 127.0.0.1 ^
  --db-port 3306 ^
  --db-user cn_opr_red ^
  --db-password YOUR_PASSWORD ^
  --db-name cn_market_red
```

Use `--replace` only when redoing the same year range after a broken partial load.

## Phase 3: V8 Compatibility Mapping History

This phase is the key addition over the old split runbook.

Do this only after Phase 2 price + board map + daily basic are available.

### 3.1 Official Tushare SW replacement source tables

```powershell
python scripts/build_tushare_sw_replacement_sources.py ^
  --start 2010-01-01 ^
  --end 2016-12-31 ^
  --srcs SW2021 SW2014 ^
  --levels L1 ^
  --replace-master ^
  --replace-members ^
  --output-dir reports/analysis/tushare_sw_replacement_sources_2010_2016
```

This builds:

- `cn_ts_sw_industry_master`
- `cn_ts_sw_industry_member_hist`

### 3.2 Local industry master

```powershell
python scripts/build_sw_industry_master.py --start 2010-01-01 --end 2016-12-31 --src SW2021
```

### 3.3 Local industry membership history

Use the script that explicitly merges board-map history and Tushare membership history:

```powershell
python scripts/build_local_industry_map_hist.py ^
  --start 2010-01-01 ^
  --end 2016-12-31 ^
  --level L1 ^
  --src SW2021 ^
  --resume ^
  --workers 4
```

This is the builder that matters for V8 local `85xxxx.SI` history.

Do **not** treat `build_sw_industry_member_hist.py` as a substitute for this step.

### 3.4 Local industry proxy daily

```powershell
python scripts/build_local_industry_proxy_daily.py ^
  --start 2010-01-01 ^
  --end 2016-12-31 ^
  --resume ^
  --workers 4
```

This builds:

- `cn_local_industry_proxy_daily`

### 3.5 V7/V8 crosswalk history

```powershell
python scripts/build_v7_v8_industry_crosswalk.py ^
  --start 2010-01-01 ^
  --end 2016-12-31 ^
  --replace ^
  --srcs SW2021 SW2014 ^
  --source-mode db ^
  --output-dir reports/analysis/sw_v7_v8_crosswalk_2010_2016
```

This builds:

- `cn_v7_v8_industry_crosswalk`

## Phase 4: Downstream Derived History

At this point, raw history, board map, local industry history, and crosswalk history are all in place.

Now rebuild the V8 derived history chain.

### 4.1 Unified derived rebuild through `v8_backfill`

```powershell
set V8_BACKFILL_START=20100101
set V8_BACKFILL_END=20161231
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
python runner.py --flag tu --tasks v8_backfill --asof 20161231
```

This covers the historical rebuild of:

- `cn_stock_fundamental_daily`
- `cn_stock_quality_score_daily`
- `cn_industry_capital_flow_daily`
- `cn_ga_stock_role_map_daily`
- `cn_stock_mainline_strength_daily`
- `cn_ga_mainline_radar_daily`
- `cn_ga_market_pulse_daily`
- `cn_mainline_lifecycle_daily`
- `cn_unified_alpha_score_daily`

### 4.2 Optional year-split daily rebuild

If you prefer the split-task style from the original runbook, use year chunks:

```powershell
set V8_DAILY_TASKS=v8_daily_market_raw,v8_daily_reference,v8_daily_audit
daily_by_year.bat 2010 2016 --refresh
```

Then, after Phase 3 is complete:

```powershell
set V8_DAILY_TASKS=v8_daily_derived_foundation,v8_daily_derived_mainline,v8_daily_derived_alpha
daily_by_year.bat 2010 2016 --refresh
```

This two-pass approach is safer than trying to build the derived layers before the local-industry and crosswalk chain exists.

## Phase 5: Latest Snapshots

These are not full-history tables. Rebuild them only after history is complete.

### 5.1 Latest V7/V8 crosswalk snapshot

```powershell
python scripts/build_v7_v8_crosswalk_latest.py ^
  --replace ^
  --output-dir reports/analysis/v7_v8_crosswalk_latest
```

### 5.2 Latest stock-level V8-to-V7 mapping snapshot

```powershell
python scripts/build_stock_v8_to_v7_sw_map_latest.py ^
  --replace ^
  --output-dir reports/analysis/stock_v8_to_v7_sw_map_latest
```

### 5.3 Latest leader SW L1 snapshot

```powershell
python -m app.tools.build_cn_stock_leader_sw_l1_latest_snap --trade-date 2016-12-30
```

Use the actual final trading day of the window for `--trade-date`.

## Recommended Order Summary

For `2010-01-01` to `2016-12-31`, the recommended order is:

1. bootstrap empty-DB schema, procedures, and GA migration if needed
2. `v8_backfill` raw price only
3. index history
4. `v8_backfill` daily basic + raw financials + SW daily
5. daily / periodic events
6. `weekly_by_year.bat 2010 2016 --refresh`
7. `build_board_member_map_full.py --resume`
8. `build_tushare_sw_replacement_sources.py`
9. `build_sw_industry_master.py`
10. `build_local_industry_map_hist.py`
11. `build_local_industry_proxy_daily.py`
12. `build_v7_v8_industry_crosswalk.py`
13. `v8_backfill` derived only
14. `build_v7_v8_crosswalk_latest.py`
15. `build_stock_v8_to_v7_sw_map_latest.py`
16. `build_cn_stock_leader_sw_l1_latest_snap`

## Incremental Re-run Rules

Default re-run strategy:

- use `weekly_by_year.bat ... --refresh`
- use `daily_by_year.bat ... --refresh`
- use `build_* --resume`
- use `v8_backfill` with feature flags to rerun only the missing phase

Use `--replace` only for:

- `build_v7_v8_industry_crosswalk.py`
- `build_v7_v8_crosswalk_latest.py`
- `build_stock_v8_to_v7_sw_map_latest.py`
- `build_board_member_map_full.py` when rebuilding a known-bad date range
- direct builder scripts that explicitly support range-scoped replace

Do not assume `runner.py` supports `--replace`; several batch wrappers accept it only for operator compatibility and convert it into `--refresh`.

## Minimal Commands By Scenario

### Scenario A: V7 DB already has schema and procedures

Run, in order:

```powershell
set V8_BACKFILL_START=20100101
set V8_BACKFILL_END=20161231
set V8_BACKFILL_INCLUDE_PRICE=1
set V8_BACKFILL_INCLUDE_FUNDAMENTAL=0
set V8_BACKFILL_INCLUDE_DAILY_BASIC=0
set V8_BACKFILL_INCLUDE_SW_DAILY=0
set V8_BACKFILL_INCLUDE_BOARD_REFRESH=0
set V8_BACKFILL_INCLUDE_DERIVED=0
python runner.py --flag tu --tasks v8_backfill --asof 20161231
```

```powershell
set V8_BACKFILL_INCLUDE_PRICE=0
set V8_BACKFILL_INCLUDE_FUNDAMENTAL=1
set V8_BACKFILL_INCLUDE_DAILY_BASIC=1
set V8_BACKFILL_INCLUDE_SW_DAILY=1
python runner.py --flag tu --tasks v8_backfill --asof 20161231
```

```powershell
weekly_by_year.bat 2010 2016 --refresh
```

```powershell
python scripts/build_board_member_map_full.py --start 2010-01-01 --end 2016-12-31 --resume --db-host 127.0.0.1 --db-port 3306 --db-user cn_opr_red --db-password YOUR_PASSWORD --db-name cn_market_red
```

```powershell
python scripts/build_local_industry_map_hist.py --start 2010-01-01 --end 2016-12-31 --level L1 --src SW2021 --resume --workers 4
python scripts/build_local_industry_proxy_daily.py --start 2010-01-01 --end 2016-12-31 --resume --workers 4
python scripts/build_tushare_sw_replacement_sources.py --start 2010-01-01 --end 2016-12-31 --srcs SW2021 SW2014 --levels L1 --replace-master --replace-members --output-dir reports/analysis/tushare_sw_replacement_sources_2010_2016
python scripts/build_v7_v8_industry_crosswalk.py --start 2010-01-01 --end 2016-12-31 --replace --srcs SW2021 SW2014 --source-mode db --output-dir reports/analysis/sw_v7_v8_crosswalk_2010_2016
```

```powershell
set V8_BACKFILL_INCLUDE_PRICE=0
set V8_BACKFILL_INCLUDE_FUNDAMENTAL=0
set V8_BACKFILL_INCLUDE_DAILY_BASIC=0
set V8_BACKFILL_INCLUDE_SW_DAILY=0
set V8_BACKFILL_INCLUDE_BOARD_REFRESH=0
set V8_BACKFILL_INCLUDE_DERIVED=1
set V8_BACKFILL_INCLUDE_CROSSWALK_LATEST=0
python runner.py --flag tu --tasks v8_backfill --asof 20161231
```

### Scenario B: Empty DB

Do Scenario A, but prepend:

1. Phase 1.1 DDL apply from `docs/DDL/README.md`
2. `python scripts/apply_ga_p0_schema_migration.py ...`
3. restore `sp_refresh_board_member_hist` and `sp_build_board_member_map`

## Final Notes

- The split daily / weekly / monthly runbook is still useful for operations, but it is not sufficient by itself for full V8 compatibility history.
- The missing historical chain is not only `cn_v7_v8_industry_crosswalk`.
- The local-industry three-table chain is also required:
  - `cn_local_industry_master`
  - `cn_local_industry_map_hist`
  - `cn_local_industry_proxy_daily`
- For historical V8 replacement validation, the official SW replacement-source tables are also required:
  - `cn_ts_sw_industry_master`
  - `cn_ts_sw_industry_member_hist`

## Future Daily / Weekly / Monthly Maintenance

For the following compatibility tables:

- `cn_local_industry_master`
- `cn_local_industry_map_hist`
- `cn_local_industry_proxy_daily`
- `cn_ts_sw_industry_master`
- `cn_ts_sw_industry_member_hist`
- `cn_v7_v8_industry_crosswalk`

the recommended future operating rhythm is:

### Daily

Daily is optional for this chain.

Recommended:

- do not rebuild `cn_local_industry_master`
- do not rebuild `cn_ts_sw_industry_master`
- do not rebuild full `cn_v7_v8_industry_crosswalk`
- only rebuild `cn_local_industry_proxy_daily` for a short recent window if your daily downstream depends on it
- allow the normal V8 pipeline to refresh `cn_v7_v8_industry_crosswalk_latest` when enabled

Example daily catch-up for proxy only:

```powershell
python scripts/build_local_industry_proxy_daily.py --start 2026-05-01 --end 2026-05-15 --resume --workers 4 --chunk-months 1 --industry-level L1
```

### Weekly

Weekly is the recommended standard cadence for the compatibility mapping chain.

Run standard weekly refresh first:

```powershell
weekly.bat --start-date 2026-05-01 --end-date 2026-05-15 --refresh
```

Then run the compatibility companion:

```powershell
weekly_reference_compatibility.bat --start-date 2026-05-01 --end-date 2026-05-15
```

This companion batch refreshes:

- `cn_local_industry_map_hist`
- `cn_local_industry_proxy_daily`
- `cn_ts_sw_industry_member_hist`
- `cn_v7_v8_industry_crosswalk`
- `cn_v7_v8_industry_crosswalk_latest`
- `cn_stock_v8_to_v7_sw_map_latest`

Optional env switches for weekly companion:

- `V8_REFCOMP_REFRESH_LOCAL_MASTER=1`
- `V8_REFCOMP_REPLACE_TS_MASTER=1`
- `V8_REFCOMP_REPLACE_TS_MEMBERS=1`
- `V8_REFCOMP_SW_SRC=SW2021`
- `V8_REFCOMP_TS_SRCS=SW2021 SW2014`

### Monthly

Monthly is the right place to refresh low-frequency master data and run a wider repair window.

Recommended monthly flow:

```powershell
monthly.bat --start-date 2026-01-01 --end-date 2026-05-15 --refresh
```

Then run the same compatibility companion on a wider window, with local/offical master refresh enabled:

```powershell
set V8_REFCOMP_REFRESH_LOCAL_MASTER=1
set V8_REFCOMP_REPLACE_TS_MASTER=1
set V8_REFCOMP_REPLACE_TS_MEMBERS=1
weekly_reference_compatibility.bat --start-date 2026-01-01 --end-date 2026-05-15
```

Or use the monthly wrapper:

```powershell
monthly_reference_compatibility.bat --start-date 2026-01-01 --end-date 2026-05-15
```

Monthly responsibilities:

- refresh `cn_local_industry_master`
- refresh `cn_ts_sw_industry_master`
- widen repair window for `cn_local_industry_map_hist`
- widen repair window for `cn_ts_sw_industry_member_hist`
- rebuild `cn_v7_v8_industry_crosswalk` over a broader recent range
- refresh latest snapshots after the broader rebuild
