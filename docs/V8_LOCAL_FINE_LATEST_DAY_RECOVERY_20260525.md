# V8 LOCAL_FINE Latest-Day Recovery

Date: 2026-05-26

## Purpose

This note explains how to move the V8 mainline signal chain forward when:

- raw market tables have reached a newer date
- but `scripts/ensure_daily_market_mainline_signal_states.py` falls back to an
  older buildable date

Typical symptom:

- `cn_stock_daily_price` / `cn_index_daily_price` are current
- but `cn_local_industry_proxy_daily` is missing on the newest raw date
- signal guard reports a fallback such as:
  - raw latest = `2026-05-25`
  - buildable latest = `2026-05-21`

## Why this happens

The V8 mainline signal chain is bounded by the minimum of:

1. latest `cn_stock_daily_price`
2. latest `cn_index_daily_price`
3. latest `cn_stock_daily_basic`
4. latest `cn_local_industry_map_hist` `LOCAL_FINE` horizon
   (`industry_level = 'L3'`)

If either `daily_basic` or `LOCAL_FINE` mapping lags raw market data, then
`cn_local_industry_proxy_daily` cannot be built for the latest raw date.

**Additional dependency discovered 2026-05-26:**

The mainline chain has a hidden dependency on `cn_board_member_map_d`:

- `sp_materialize_leader_score` (called by `v8_stock_basic`) queries
  `cn_board_member_map_d WHERE sector_type='INDUSTRY'` to compute leader scores.
- If `cn_board_member_map_d` has no data for recent dates, leader scores silently
  skip those dates, which cascades into empty `cn_ga_stock_role_map_daily` →
  `cn_stock_mainline_strength_daily` → `cn_ga_mainline_radar_daily` →
  `cn_ga_market_pulse_daily`.
- Therefore, refreshing `cn_board_member_map_d` via `build_board_map_by_year.py`
  is a prerequisite before re-running `v8_stock_basic` and the mainline chain.

## Current V8 semantic reminder

- `cn_local_industry_map_hist.industry_level = 'L3'` = `LOCAL_FINE`
- `cn_local_industry_proxy_daily.industry_level = 'L1'` = legacy physical label
  for the same `LOCAL_FINE` production layer

## Minimal recovery order

### Step 1. Refresh daily basic to the target raw window

Preferred runner path:

```bat
set STOCK_BASIC_LOOKBACK_DAYS=7
python runner.py --tasks v8_stock_basic --start-date 2026-05-19 --end-date 2026-05-25 --refresh
```

If you want a wider safety window, increase `STOCK_BASIC_LOOKBACK_DAYS`.

**Note:** `v8_stock_basic` internally calls `sp_materialize_leader_score` which
materializes `cn_stock_leader_score_daily`. This stored procedure depends on
`cn_board_member_map_d` having `INDUSTRY` records for the target dates. If
`cn_board_member_map_d` is stale, leader scores will silently skip recent dates.
See Step 1.5 below.

### Step 1.5. Refresh board member map (prerequisite for leader scores)

If `cn_board_member_map_d` is missing recent dates, run this **before** Step 1:

```bat
python app/tools/build_board_map_by_year.py --start 2026-05-19 --end 2026-05-25
```

This calls `sp_build_board_member_map` by year chunks to ensure
`cn_board_member_map_d` covers the full target window. Without this,
`sp_materialize_leader_score` will silently skip dates with no board membership
data, causing the entire mainline chain to be empty for those dates.

### Step 2. Refresh `LOCAL_FINE` mapping horizon

Required V8 layer:

```bat
python scripts/build_local_industry_map_hist.py --start 2026-01-01 --end 2026-05-25 --level L3 --src SW2021 --resume --workers 4
```

If you want to force re-evaluation of the recent range:

```bat
python scripts/build_local_industry_map_hist.py --start 2026-05-01 --end 2026-05-25 --level L3 --src SW2021 --force --workers 4
```

### Step 3. Rebuild local proxy for the same recent window

```bat
python scripts/build_local_industry_proxy_daily.py --start 2026-05-19 --end 2026-05-25 --chunk-days 5
```

### Step 4. Re-run the full mainline derived chain

After the board map, stock basic, LOCAL_FINE mapping, and proxy are current,
re-run the mainline derived chain to fill `cn_ga_stock_role_map_daily`,
`cn_stock_mainline_strength_daily`, `cn_ga_mainline_radar_daily`,
`cn_ga_market_pulse_daily`, and `cn_mainline_lifecycle_daily`:

```bat
REM Step 4a: Re-run derived foundation (role_map only) — skip stock_fundamental_daily,
REM stock_quality_score, and industry_capital_flow since they are not affected by
REM board map refresh. This eliminates ~5 minutes of redundant computation.
set V8_SKIP_STOCK_FUNDAMENTAL_DAILY=1
set V8_SKIP_STOCK_QUALITY_SCORE=1
set V8_SKIP_INDUSTRY_CAPITAL_FLOW=1
python runner.py --tasks v8_daily_derived_foundation --start-date 2026-05-19 --end-date 2026-05-25 --refresh

REM Step 4b: Re-run derived mainline chain (full — all builders are affected by role_map changes)
python runner.py --tasks v8_daily_derived_mainline --start-date 2026-05-19 --end-date 2026-05-25 --refresh
```

### Step 5. Re-run the signal guard

```bat
python scripts/ensure_daily_market_mainline_signal_states.py
```

Expected behavior:

- if both `daily_basic` and `LOCAL_FINE` mapping now cover the target date, the
  guard should advance automatically
- if not, the guard will still fall back to the latest buildable date and print
  the limiting horizon

## Fast diagnosis queries

```sql
SELECT MAX(trade_date) FROM cn_stock_daily_price;
SELECT MAX(trade_date) FROM cn_index_daily_price;
SELECT MAX(trade_date) FROM cn_stock_daily_basic;
SELECT MAX(COALESCE(out_date, in_date)) FROM cn_local_industry_map_hist WHERE industry_level = 'L3';
SELECT MAX(trade_date) FROM cn_local_industry_proxy_daily;
SELECT MAX(trade_date) FROM cn_board_member_map_d;
SELECT MAX(trade_date) FROM cn_stock_leader_score_daily;
SELECT MAX(trade_date) FROM cn_ga_stock_role_map_daily;
SELECT MAX(trade_date) FROM cn_stock_mainline_strength_daily;
SELECT MAX(trade_date) FROM cn_ga_mainline_radar_daily;
SELECT MAX(trade_date) FROM cn_ga_market_pulse_daily;
```

**Note:** The LOCAL_FINE horizon query now uses `MAX(COALESCE(out_date, in_date))`
instead of `MAX(out_date)`. `out_date` is the date a stock *leaves* an industry;
stocks still in their industry have `NULL` out_date, which would incorrectly pull
the max down. `COALESCE(out_date, in_date)` falls back to `in_date` when
`out_date` is NULL, giving the correct latest horizon.

## Operator rule of thumb

- If `daily_basic` is behind, fix `v8_stock_basic` first.
- If `cn_board_member_map_d` is behind, run `build_board_map_by_year.py` first
  (this is a prerequisite for leader scores).
- If `LOCAL_FINE` map is behind, fix `build_local_industry_map_hist.py --level L3`.
- Only after those three are current does it make sense to expect the local proxy
  and mainline signal chain to reach the newest raw market date.
- If the mainline chain still lags after Steps 1-3, run Step 4
  (`v8_daily_derived_mainline`) explicitly to fill the gap.
