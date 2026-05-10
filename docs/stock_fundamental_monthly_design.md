# Stock Fundamental Monthly

This task adds a monthly historical fundamental pipeline for A-shares.

## Scope

- month-end valuation / market-cap snapshots from `2008-01-01`
- quarterly income statement history from `2008-01-01`
- quarterly balance-sheet history from `2008-01-01`
- quarterly financial-quality history from `2008-01-01`
- `tushare` first
- free-source fallback via `akshare`

## Minimum Financial Set

This monthly pipeline is optimized for four baseline quality checks:

- `eps > 0`
- revenue growth: prefer `or_yoy`, fallback `tr_yoy`, then `q_sales_yoy`
- `debt_to_eqt < 2`
- `grossprofit_margin` as optional quality improvement

## Objects

- `cn_stock_monthly_basic`
- `cn_stock_income`
- `cn_stock_balancesheet`
- `cn_stock_fina_indicator`
- `cn_stock_fundamental_quality_v1`
- `cn_stock_fundamental_quality_hist_v1`
- `cn_stock_fundamental_quality_snap`
- `cn_stock_working_capital_alert_v1`
- `cn_fundamental_quality_param_t`
- `app.tools.sync_cn_stock_fundamental_monthly`
- `app.tasks.stock_fundamental_monthly_task`
- `app.tasks.stock_quality_snapshot_task`
- `app.tasks.stock_working_capital_alert_task`

## Financial Object Summary

### Newly added in this enhancement

Tables:

- `cn_stock_balancesheet`

Views:

- `cn_stock_working_capital_alert_v1`

### Existing objects expanded or refreshed by this module

Tables:

- `cn_stock_monthly_basic`
- `cn_stock_income`
- `cn_stock_fina_indicator`
- `cn_stock_fundamental_quality_snap`

Views:

- `cn_stock_fundamental_quality_v1`
- `cn_stock_fundamental_quality_hist_v1`

### Current role of each object

- `cn_stock_monthly_basic`
  - monthly valuation / market-cap snapshot
- `cn_stock_income`
  - quarterly income statement history
- `cn_stock_balancesheet`
  - quarterly balance-sheet history
- `cn_stock_fina_indicator`
  - quarterly financial indicators
- `cn_stock_fundamental_quality_v1`
  - latest quality view
- `cn_stock_fundamental_quality_hist_v1`
  - historical as-of quality view
- `cn_stock_fundamental_quality_snap`
  - materialized snapshot of the historical quality view
- `cn_stock_working_capital_alert_v1`
  - latest receivables / inventory anomaly view

## Task Name

Use:

```powershell
python -m app.cli --tasks stock_fundamental --asof latest
```

Refresh only the working-capital anomaly view:

```powershell
python -m app.cli --tasks stock_working_capital --asof latest
```

Rebuild only the quality snapshot:

```powershell
python -m app.cli --tasks stock_quality_snapshot --asof latest
```

For first full backfill:

```powershell
set STOCK_FUNDAMENTAL_MONTHLY_FORCE=1
set STOCK_FUNDAMENTAL_MONTHLY_HISTORY_START=20080101
python -m app.cli --tasks stock_fundamental --asof latest
```

For full backfill from `2010-01-01`:

```powershell
set STOCK_FUNDAMENTAL_MONTHLY_FORCE=1
set STOCK_FUNDAMENTAL_MONTHLY_HISTORY_START=20100101
set STOCK_FUNDAMENTAL_MONTHLY_FULL_REBUILD=1
python -m app.cli --tasks stock_fundamental --asof latest
```

Independent mode:

```powershell
python -m app.tools.sync_cn_stock_fundamental_monthly --provider tushare --start 2008-01-01 --end 2026-04-18
```

## Scheduling

Default policy:

- runs once per month
- default run day: day `1`
- skips automatically on other days unless forced

Env vars:

- `STOCK_FUNDAMENTAL_MONTHLY_ENABLED=1`
- `STOCK_FUNDAMENTAL_MONTHLY_FORCE=1`
- `STOCK_FUNDAMENTAL_MONTHLY_MONTHDAY=1`
- `STOCK_FUNDAMENTAL_MONTHLY_PROVIDER=auto`
- `STOCK_FUNDAMENTAL_MONTHLY_HISTORY_START=20080101`
- `STOCK_FUNDAMENTAL_MONTHLY_FULL_REBUILD=0`
- `STOCK_FUNDAMENTAL_MONTHLY_CALENDAR_SOURCE=price`
- `STOCK_FUNDAMENTAL_MONTHLY_BASIC_SOURCE_LABEL=tushare_monthly_basic`
- `STOCK_FUNDAMENTAL_MONTHLY_INCOME_SOURCE_LABEL=tushare_income`
- `STOCK_FUNDAMENTAL_MONTHLY_BALANCE_SOURCE_LABEL=tushare_balancesheet`
- `STOCK_FUNDAMENTAL_MONTHLY_FINA_SOURCE_LABEL=tushare_fina_indicator`
- `STOCK_FUNDAMENTAL_MONTHLY_AKSHARE_WORKERS=8`
- `STOCK_FUNDAMENTAL_MONTHLY_AKSHARE_TIMEOUT=15`
- `STOCK_FUNDAMENTAL_MONTHLY_SKIP_QUALITY_SNAPSHOT=0`
- `STOCK_WORKING_CAPITAL_ALERT_ENABLED=1`

## Notes

- `cn_stock_monthly_basic` uses the last available trade date of each month.
- `cn_stock_income` stores the official profit statement history and is the primary source for `n_income_attr_p`.
- `cn_stock_balancesheet` stores official Tushare balance-sheet rows including `accounts_receiv` and `inventories`.
- `cn_stock_fina_indicator` stores quarterly report-period history.
- fetch-shape rule:
  - `monthly_basic` is `date-first` because `daily_basic(trade_date=...)` is a market snapshot API
  - `income` / `balancesheet` / `fina_indicator` / `cashflow` are `symbol-first` because the primary API entrance is `ts_code`
  - even in incremental mode driven by `cn_event_disclosure_date`, the loader groups disclosure dates by `symbol`, fetches one compact date window per `symbol`, then filters locally by disclosure dates
  - historical backfill with `--by-symbol` keeps the same `symbol-first` shape and expands only the date window
- both tables keep a `raw_payload` copy of the full provider row so future factors can be added without re-pulling old periods
- recommended minimum fields for downstream strategies:
  - `cn_stock_income.n_income_attr_p`
  - `cn_stock_income.ann_date`
  - `cn_stock_income.end_date`
  - `cn_stock_balancesheet.accounts_receiv`
  - `cn_stock_balancesheet.inventories`
  - `cn_stock_fina_indicator.q_profit_yoy`
  - `cn_stock_fina_indicator.netprofit_yoy`
- `cn_stock_fundamental_quality_v1` exposes latest screen flags:
  - `pass_eps_positive`
  - `pass_revenue_growth_5`
  - `pass_revenue_growth_10`
  - `pass_debt_to_eqt_lt_2`
  - `pass_gross_margin_positive`
  - `quality_pass_core`
  - `quality_pass_with_margin`
- `cn_stock_fundamental_quality_hist_v1` is the as-of historical monthly view
- `cn_stock_fundamental_quality_snap` materializes that historical view for faster reads
- `cn_stock_working_capital_alert_v1` is the latest anomaly view for receivables / inventory growth-vs-revenue checks
- thresholds are driven by the active row in `cn_fundamental_quality_param_t`
- `akshare` fallback is best-effort:
  - monthly basic is reliable for latest snapshot fallback
  - financial indicators depend on free endpoint availability and field shape
  - balance-sheet history currently uses Tushare only
- `STOCK_FUNDAMENTAL_MONTHLY_FORCE=1` means "run today even if not on the scheduled day"
- `STOCK_FUNDAMENTAL_MONTHLY_FULL_REBUILD=1` means "ignore existing max dates and rescan from HISTORY_START"
- `STOCK_FUNDAMENTAL_MONTHLY_SKIP_QUALITY_SNAPSHOT=1` means "skip rebuilding cn_stock_fundamental_quality_snap if temp-table pressure is too high"
- console progress now prints launch banner plus rolling progress / ETA so long backfills are observable

## Working-Capital Alert View

Current rules:

- `accounts_receiv_yoy_pct > revenue_yoy_pct`
- `inventories_yoy_pct > revenue_yoy_pct`

Key fields:

- `accounts_receiv_yoy_pct`
- `inventories_yoy_pct`
- `revenue_yoy_pct`
- `receiv_growth_gap_pct`
- `inventory_growth_gap_pct`
- `working_capital_alert_score`
- `working_capital_alert_level`

The monthly fundamental task refreshes this view after statement sync, and it can also be refreshed independently by `stock_working_capital`.

Recommended operation:

- normal run: `stock_fundamental` only
- quality snapshot only: `stock_quality_snapshot`
- view-only rebuild: `stock_working_capital`

## Export

Export latest picks:

```powershell
python -m app.tools.export_fundamental_quality_latest --mode core --limit 200
```

Output is written to `audit_reports/`.
