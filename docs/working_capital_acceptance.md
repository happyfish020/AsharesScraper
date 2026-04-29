# Working Capital Acceptance

## Goal

Use this checklist after `stock_fundamental` full backfill completes to verify that:

- `cn_stock_balancesheet` exists and is populated
- `cn_stock_working_capital_alert_v1` exists and is queryable
- receivables / inventory anomaly fields are available
- `ann_date` is present for later historical visibility checks

Recommended run order before acceptance:

```powershell
$env:STOCK_FUNDAMENTAL_MONTHLY_HISTORY_START="20100101"
$env:STOCK_FUNDAMENTAL_MONTHLY_FORCE="1"
$env:STOCK_FUNDAMENTAL_MONTHLY_FULL_REBUILD="1"
python runner.py --tasks stock_fundamental --asof latest
```

## 1. Object Existence

Check whether the table and view exist:

```sql
SELECT table_name, table_type
FROM information_schema.tables
WHERE table_schema = 'cn_market'
  AND table_name IN (
    'cn_stock_balancesheet',
    'cn_stock_working_capital_alert_v1'
  )
ORDER BY table_name;
```

Expected:

- `cn_stock_balancesheet` exists
- `cn_stock_working_capital_alert_v1` exists

## 2. Balance Sheet Columns

Check core columns in `cn_stock_balancesheet`:

```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'cn_market'
  AND table_name = 'cn_stock_balancesheet'
  AND column_name IN (
    'symbol',
    'end_date',
    'ann_date',
    'f_ann_date',
    'accounts_receiv',
    'inventories',
    'notes_receiv',
    'total_assets',
    'total_liab'
  )
ORDER BY ordinal_position;
```

Expected:

- all listed columns are present

## 3. Working-Capital View Columns

Check core columns in `cn_stock_working_capital_alert_v1`:

```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'cn_market'
  AND table_name = 'cn_stock_working_capital_alert_v1'
  AND column_name IN (
    'symbol',
    'end_date',
    'ann_date',
    'accounts_receiv_yoy_pct',
    'inventories_yoy_pct',
    'revenue_yoy_pct',
    'receiv_growth_gap_pct',
    'inventory_growth_gap_pct',
    'working_capital_alert_score',
    'working_capital_alert_level'
  )
ORDER BY ordinal_position;
```

Expected:

- all listed columns are present

## 4. Balance Sheet Coverage

Check overall table coverage:

```sql
SELECT
    COUNT(*) AS row_cnt,
    COUNT(DISTINCT symbol) AS symbol_cnt,
    MIN(end_date) AS min_end_date,
    MAX(end_date) AS max_end_date,
    MIN(ann_date) AS min_ann_date,
    MAX(ann_date) AS max_ann_date
FROM cn_stock_balancesheet;
```

Review points:

- `min_end_date` should move back toward the historical target window
- `max_end_date` should reach the latest loaded report period
- `symbol_cnt` should be broad enough for the active A-share universe

## 5. Core Field Non-Null Check

Verify that key balance-sheet fields actually contain values:

```sql
SELECT
    COUNT(*) AS row_cnt,
    SUM(CASE WHEN accounts_receiv IS NOT NULL THEN 1 ELSE 0 END) AS accounts_receiv_not_null,
    SUM(CASE WHEN inventories IS NOT NULL THEN 1 ELSE 0 END) AS inventories_not_null,
    SUM(CASE WHEN notes_receiv IS NOT NULL THEN 1 ELSE 0 END) AS notes_receiv_not_null
FROM cn_stock_balancesheet;
```

Expected:

- `accounts_receiv_not_null` is materially above zero
- `inventories_not_null` is materially above zero

## 6. Raw Sample Rows

Inspect a few recent rows:

```sql
SELECT
    symbol,
    end_date,
    ann_date,
    accounts_receiv,
    inventories,
    notes_receiv,
    total_assets,
    total_liab
FROM cn_stock_balancesheet
ORDER BY end_date DESC, symbol
LIMIT 20;
```

Use this to confirm:

- field values look numeric and reasonable
- dates are correctly populated

## 7. Working-Capital View Coverage

Check whether the view returns a usable result set:

```sql
SELECT
    COUNT(*) AS row_cnt,
    COUNT(DISTINCT symbol) AS symbol_cnt,
    MIN(end_date) AS min_end_date,
    MAX(end_date) AS max_end_date
FROM cn_stock_working_capital_alert_v1;
```

Expected:

- row count is above zero
- symbol count is above zero

## 8. Alert Level Distribution

Inspect the distribution of current alert levels:

```sql
SELECT
    working_capital_alert_level,
    COUNT(*) AS cnt
FROM cn_stock_working_capital_alert_v1
GROUP BY working_capital_alert_level
ORDER BY cnt DESC;
```

Expected:

- returns at least `normal`
- `watch` / `high` may be present depending on data completeness

## 9. High / Watch Review

Inspect the most suspicious names:

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
ORDER BY working_capital_alert_score DESC, end_date DESC, symbol
LIMIT 50;
```

Use this to confirm:

- anomaly values are populated
- score / level mapping looks reasonable

## 10. Single-Stock Sanity Check

Example using `688981`:

```sql
SELECT
    symbol,
    end_date,
    ann_date,
    accounts_receiv_yoy_pct,
    inventories_yoy_pct,
    revenue_yoy_pct,
    receiv_growth_gap_pct,
    inventory_growth_gap_pct,
    working_capital_alert_score,
    working_capital_alert_level
FROM cn_stock_working_capital_alert_v1
WHERE symbol = '688981'
ORDER BY end_date DESC;
```

Use this to verify:

- one stock returns a stable history
- gaps are calculated as expected

## 11. `ann_date` Availability

Check whether later historical visibility filtering is feasible:

```sql
SELECT
    COUNT(*) AS row_cnt,
    SUM(CASE WHEN ann_date IS NOT NULL THEN 1 ELSE 0 END) AS ann_date_not_null
FROM cn_stock_balancesheet;
```

Expected:

- `ann_date_not_null` should be materially above zero

## 12. Missing `ann_date` Review

Inspect rows missing announcement dates:

```sql
SELECT
    symbol,
    end_date,
    ann_date,
    f_ann_date,
    accounts_receiv,
    inventories
FROM cn_stock_balancesheet
WHERE ann_date IS NULL
ORDER BY end_date DESC, symbol
LIMIT 50;
```

Use this to decide:

- whether the current dataset is sufficient for strict `ann_date <= screen_date` history rules

## Acceptance Summary

### Structure passes if

- `cn_stock_balancesheet` exists
- `cn_stock_working_capital_alert_v1` exists
- required columns exist in both objects

### Data passes if

- `cn_stock_balancesheet` has broad historical coverage
- `accounts_receiv` and `inventories` are populated for a meaningful share of rows
- the working-capital view returns non-zero rows

### Business passes if

- `working_capital_alert_level` distribution looks reasonable
- sample names return interpretable anomaly values
- single-stock checks are stable

### Time-visibility readiness passes if

- `ann_date` is sufficiently populated in `cn_stock_balancesheet`
- missing `ann_date` rows are limited and explainable
