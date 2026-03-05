# Board Membership Refresh Playbook (DB-first)

## Scope

This playbook covers board/concept membership refresh in MySQL with:

1. Historical base tables with validity window (`valid_from`, `valid_to`)
2. Daily expanded mapping table for strategy joins

Objects are created in database `cn_market`.

## Critical Constraint For Historical Backtest

For historical ranges (for example `2000-2005`), do not use `cn_board_industry_cons` / `cn_board_concept_cons` as historical mapping source.

Reason:

1. `cn_board_*_cons` is snapshot-like in this environment (often only near latest `ASOF_DATE`, such as around `2026-01-09`).
2. Old-year `ASOF_DATE` coverage is incomplete, so symbol-to-sector mapping can be wrong or empty.

Historical backtest mapping must use:

1. `cn_board_member_map_d(trade_date, sector_type, sector_id, symbol)` only.
2. If map coverage is missing, rebuild from `cn_board_*_member_hist` first, then rerun downstream backfill.

Pre-flight checks:

```sql
-- map coverage in target range
SELECT MIN(trade_date) AS min_d, MAX(trade_date) AS max_d, COUNT(DISTINCT trade_date) AS d_cnt
FROM cn_board_member_map_d
WHERE trade_date BETWEEN :D1 AND :D2;

-- cons table date span (to confirm snapshot-like behavior)
SELECT 'industry_cons' AS src, MIN(asof_date) AS min_d, MAX(asof_date) AS max_d, COUNT(DISTINCT asof_date) AS d_cnt
FROM cn_board_industry_cons
UNION ALL
SELECT 'concept_cons', MIN(asof_date), MAX(asof_date), COUNT(DISTINCT asof_date)
FROM cn_board_concept_cons;
```

## Created Objects

### Tables

1. `cn_board_concept_member_hist`
2. `cn_board_industry_member_hist`
3. `cn_board_member_map_d`
4. `cn_board_concept_member_stg`
5. `cn_board_industry_member_stg`

### Stored Procedures

1. `sp_refresh_board_member_hist(p_asof_date, p_source, p_apply_concept, p_apply_industry)`
2. `sp_build_board_member_map(p_start_date, p_end_date)`
3. `sp_refresh_sector_eod_hist(p_from, p_to, p_top_pct, p_breadth_min)` (now map-driven)

## Data Model

### 1) History base tables (SCD2-like)

- Concept:
  - key: `(concept_id, symbol, valid_from)`
  - open interval row: `valid_to IS NULL`
- Industry:
  - key: `(board_id, symbol, valid_from)`
  - open interval row: `valid_to IS NULL`

### 2) Daily mapping table

- `cn_board_member_map_d(trade_date, sector_type, sector_id, symbol)`
- built from history validity windows and trading dates in `cn_stock_daily_price`

## Refresh Workflow

### Step A: Load external snapshot into staging

For target date `:ASOF_DATE`, load external data (for example Tushare) into:

1. `cn_board_concept_member_stg(asof_date, concept_id, symbol, source)`
2. `cn_board_industry_member_stg(asof_date, board_id, symbol, source)`

Suggested pre-clean for rerun:

```sql
DELETE FROM cn_board_concept_member_stg WHERE asof_date = :ASOF_DATE;
DELETE FROM cn_board_industry_member_stg WHERE asof_date = :ASOF_DATE;
```

### Step B: Refresh validity history

```sql
CALL sp_refresh_board_member_hist(:ASOF_DATE, 'tushare', 1, 1);
```

Behavior:

1. close open rows not present in current staging snapshot (`valid_to = asof_date - 1`)
2. insert new open rows for newly appeared members (`valid_from = asof_date`, `valid_to = NULL`)

Safety rule:

- if staging is empty for enabled domain (concept/industry), procedure throws error and exits.

### Step C: Build daily mapping for range

```sql
CALL sp_build_board_member_map(:D1, :D2);
```

Behavior:

1. delete existing map rows in `[D1, D2]`
2. rebuild from history validity windows and price trading dates

## Incremental Run (one date)

```sql
-- 1) load staging rows for date d
-- 2) refresh history
CALL sp_refresh_board_member_hist('2026-02-28', 'tushare', 1, 1);

-- 3) build map for same date
CALL sp_build_board_member_map('2026-02-28', '2026-02-28');
```

## One-command Program Run

After staging is loaded for asof date, run:

```bash
python runner.py --tasks board --asof latest
```

Or run a fixed date:

```bash
python runner.py --tasks board --asof 20260228
```

Environment switches:

1. `BOARD_MEMBERSHIP_SOURCE` default `tushare`
2. `BOARD_APPLY_CONCEPT` default `1`
3. `BOARD_APPLY_INDUSTRY` default `1`

## Historical Backfill From Tushare (SW L3 Industry)

Script:

- `app/tools/backfill_sw_l3_history_from_tushare.py`

Usage:

```bash
set TUSHARE_TOKEN=YOUR_TOKEN
python -m app.tools.backfill_sw_l3_history_from_tushare --start 2000-01-01 --end 2022-12-31 --src SW2021 --source-label tushare_sw_l3
```

Notes:

1. Script first tries `index_member_all`; if token cannot access that dataset, it falls back to `index_member`.
2. Script writes to `cn_board_industry_member_hist` and then rebuilds `cn_board_member_map_d` for the range.
3. This script is for INDUSTRY mapping. Concept history needs a separate source chain.

## Historical Backfill From Tushare (Concept)

Script:

- `app/tools/backfill_concept_history_from_tushare.py`

Usage:

```bash
set TUSHARE_TOKEN=YOUR_TOKEN
python -m app.tools.backfill_concept_history_from_tushare --start 2000-01-01 --end 2022-12-31 --source-label tushare_concept
```

Notes:

1. Uses `concept` + `concept_detail` and writes `in_date/out_date` into `cn_board_concept_member_hist`.
2. Rebuilds `cn_board_member_map_d` for the selected range after import.
3. Available concept history starts from source availability (not necessarily 2000).

## Downstream Logic Update (Energy / Signal / Rotation)

Updated objects:

1. `cn_board_industry_eod_agg_v`
2. `cn_board_concept_eod_agg_v`
3. `sp_refresh_sector_eod_hist`

Current behavior:

1. Board aggregation no longer uses latest snapshot-only `cn_board_*_cons`.
2. Aggregation uses `cn_board_member_map_d(trade_date, sector_type, sector_id, symbol)`.
3. `sp_refresh_sector_eod_hist` deletes and rebuilds requested date range from map data, so old stale rows are removed.
4. Existing `SP_ROTATION_DAILY_REFRESH -> sp_refresh_sector_eod_hist -> ranked/signal` chain now naturally consumes history mapping by date.

## Backfill Run (date range)

Repeat Step A and Step B per snapshot date, then rebuild map in one shot:

```sql
CALL sp_build_board_member_map('2020-01-01', '2026-02-28');
```

## Validation SQL

### Latest history open rows

```sql
SELECT 'CONCEPT' AS t, COUNT(*) AS open_rows
FROM cn_board_concept_member_hist
WHERE valid_to IS NULL
UNION ALL
SELECT 'INDUSTRY', COUNT(*)
FROM cn_board_industry_member_hist
WHERE valid_to IS NULL;
```

### Daily map counts

```sql
SELECT trade_date, sector_type, COUNT(*) AS n
FROM cn_board_member_map_d
WHERE trade_date BETWEEN '2026-02-24' AND '2026-02-28'
GROUP BY trade_date, sector_type
ORDER BY trade_date, sector_type;
```

### Coverage check against price trading days

```sql
SELECT p.trade_date
FROM (
  SELECT DISTINCT trade_date
  FROM cn_stock_daily_price
  WHERE trade_date BETWEEN '2026-01-01' AND '2026-02-28'
) p
LEFT JOIN (
  SELECT DISTINCT trade_date
  FROM cn_board_member_map_d
) m ON m.trade_date = p.trade_date
WHERE m.trade_date IS NULL
ORDER BY p.trade_date;
```
