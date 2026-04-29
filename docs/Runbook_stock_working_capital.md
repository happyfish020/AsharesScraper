# Stock Working Capital Runbook

## Goal

Build and refresh the A-share working-capital anomaly module in MySQL.

This module depends on three financial statement objects:

- `cn_stock_balancesheet`
- `cn_stock_income`
- `cn_stock_fina_indicator`

and produces:

- `cn_stock_working_capital_alert_v1`

## Object Summary

### Newly added in this enhancement

Tables:

- `cn_stock_balancesheet`

Views:

- `cn_stock_working_capital_alert_v1`

### Existing upstream dependencies

- `cn_stock_income`
- `cn_stock_fina_indicator`
- `cn_stock_monthly_basic`

### Refresh behavior

- `stock_fundamental`
  - refreshes raw financial tables
  - refreshes quality views / snapshot
  - refreshes `cn_stock_working_capital_alert_v1`
- `stock_working_capital`
  - refreshes only `cn_stock_working_capital_alert_v1`

## Task Entry Points

### Recommended: one command

Use this as the normal entry point:

```powershell
python runner.py --tasks stock_fundamental --asof latest
```

Why this is enough:

- `stock_fundamental` pulls / upserts financial statements
- at the end of the task, it also refreshes `cn_stock_working_capital_alert_v1`
- so in normal runs you do not need to run `stock_working_capital` again

### Optional: refresh only the alert view

Use this only when statement tables are already up to date and you want to rebuild just the alert view:

```powershell
python runner.py --tasks stock_working_capital --asof latest
```

### Optional: rebuild only the quality snapshot

Use this when `cn_stock_fundamental_quality_snap` needs to be rebuilt separately:

```powershell
python runner.py --tasks stock_quality_snapshot --asof latest
```

Notes:

- this task rebuilds `cn_stock_fundamental_quality_snap`
- it uses monthly-batch insertion via `month_key`
- it avoids the old one-shot full-table insert pattern

## Full Backfill From 2010 To Latest

If you want to backfill `cn_stock_balancesheet` from `2010-01-01` to latest, use one command:

```powershell
$env:STOCK_FUNDAMENTAL_MONTHLY_HISTORY_START="20100101"
$env:STOCK_FUNDAMENTAL_MONTHLY_FORCE="1"
$env:STOCK_FUNDAMENTAL_MONTHLY_FULL_REBUILD="1"
python runner.py --tasks stock_fundamental --asof latest
```

Notes:

- `stock_fundamental` is the task that actually pulls Tushare statements
- `stock_fundamental` already refreshes `cn_stock_working_capital_alert_v1` at the end
- `stock_working_capital` does not pull raw statements
- use `stock_working_capital` only for view-only rebuilds after data repair or rule changes

## Current Detection Rules

The current version flags a stock when:

- `accounts_receiv_yoy_pct > revenue_yoy_pct`
- `inventories_yoy_pct > revenue_yoy_pct`

Scoring:

- `2` -> `high`
- `1` -> `watch`
- `0` -> `normal`

## Main Output Fields

- `symbol`
- `end_date`
- `accounts_receiv`
- `prev_year_accounts_receiv`
- `accounts_receiv_yoy_pct`
- `inventories`
- `prev_year_inventories`
- `inventories_yoy_pct`
- `revenue_yoy_pct`
- `receiv_growth_gap_pct`
- `inventory_growth_gap_pct`
- `working_capital_alert_score`
- `working_capital_alert_level`

## Query Examples

### Latest high / watch list

```sql
SELECT
    symbol,
    end_date,
    accounts_receiv_yoy_pct,
    inventories_yoy_pct,
    revenue_yoy_pct,
    receiv_growth_gap_pct,
    inventory_growth_gap_pct,
    working_capital_alert_score,
    working_capital_alert_level
FROM cn_stock_working_capital_alert_v1
WHERE working_capital_alert_level IN ('high', 'watch')
ORDER BY working_capital_alert_score DESC, end_date DESC, symbol;
```

### One stock check

```sql
SELECT *
FROM cn_stock_working_capital_alert_v1
WHERE symbol = '688981';
```

## Environment Variables

### Full statement sync

- `STOCK_FUNDAMENTAL_MONTHLY_ENABLED=1`
- `STOCK_FUNDAMENTAL_MONTHLY_FORCE=1`
- `STOCK_FUNDAMENTAL_MONTHLY_HISTORY_START=20100101`
- `STOCK_FUNDAMENTAL_MONTHLY_FULL_REBUILD=1`
- `STOCK_FUNDAMENTAL_MONTHLY_PROVIDER=auto`
- `STOCK_FUNDAMENTAL_MONTHLY_BALANCE_SOURCE_LABEL=tushare_balancesheet`
- `STOCK_FUNDAMENTAL_MONTHLY_INCOME_SOURCE_LABEL=tushare_income`
- `STOCK_FUNDAMENTAL_MONTHLY_FINA_SOURCE_LABEL=tushare_fina_indicator`
- `STOCK_FUNDAMENTAL_MONTHLY_SKIP_QUALITY_SNAPSHOT=0`

### Alert refresh

- `STOCK_WORKING_CAPITAL_ALERT_ENABLED=1`

## Operational Notes

- `cn_stock_balancesheet` is currently sourced from Tushare
- `cn_stock_working_capital_alert_v1` is a view, not a snapshot table
- if you do a first-time historical backfill, expect `stock_fundamental` to take much longer than `stock_working_capital`
- if historical prior-year rows are missing, YOY fields in the alert view will be `NULL`
- `STOCK_FUNDAMENTAL_MONTHLY_FORCE=1` only removes the run-day restriction
- `STOCK_FUNDAMENTAL_MONTHLY_FULL_REBUILD=1` is the switch that forces rescan from `STOCK_FUNDAMENTAL_MONTHLY_HISTORY_START`
- if `cn_stock_fundamental_quality_snap` rebuild hits MySQL temp-table limits, set `STOCK_FUNDAMENTAL_MONTHLY_SKIP_QUALITY_SNAPSHOT=1` and rerun the financial load first

## Recommended Daily / Monthly Usage

### Monthly financial refresh

```powershell
python runner.py --tasks stock_fundamental --asof latest
```

### Ad-hoc alert refresh after data repair or DDL change

```powershell
python runner.py --tasks stock_working_capital --asof latest
```
