# DECISION_SUPPORT Rotation Recovery And Operations Runbook

Date: 2026-05-18

## Purpose

This runbook documents how to recover and operate the industry-rotation data chain used by `DECISION_SUPPORT`.

It is written for the specific failure mode where:

- `cn_board_industry_member_hist` lost most of its rows
- `cn_board_member_map_d` became incomplete or stale
- downstream rotation history no longer matches prior report behavior

The main goal is to restore a stable Shenwan Level-1 (`801%.SI`) industry mapping chain and the downstream rotation tables derived from that mapping.

## Critical Rule

For historical rotation analysis, do not treat snapshot-like `cn_board_industry_cons` / `cn_board_concept_cons` as the historical source of truth.

Historical mapping must be rebuilt from:

1. `cn_board_industry_member_hist`
2. `cn_board_member_map_d`

Then downstream rotation tables must be rebuilt from the map.

## Core Tables

### Mapping history layer

- `cn_board_industry_member_hist`
- `cn_board_member_map_d`

### Rotation analysis layer

- `cn_sector_eod_hist_t`
- `cn_sector_rotation_ranked_t`
- `cn_sector_rotation_signal_t`

### Optional backtest / snapshot layer

- `cn_sector_rot_bt_daily_t`
- `cn_sector_rot_pos_daily_t`
- `cn_rotation_entry_snap_t`
- `cn_rotation_holding_snap_t`
- `cn_rotation_exit_snap_t`

## Failure Pattern

This runbook is for cases like:

- `cn_board_industry_member_hist` dropped from thousands of rows to tens of rows
- `DECISION_SUPPORT` industry rotation results changed sharply after DB recovery
- industry membership history no longer covers `2010-01-04` onward

In this state, report logic may still run, but the underlying sector mapping is wrong, so the report result is not trustworthy.

## Recovery Sequence

Always recover in this order:

1. restore `cn_board_industry_member_hist`
2. rebuild `cn_board_member_map_d`
3. rebuild rotation three-table history
4. optionally rebuild rotation backtest tables
5. optionally refresh latest snapshots
6. rerun `DECISION_SUPPORT`

Do not start from the report layer.

## Step 1: Restore Industry Member History

Use the historical Tushare SW loader:

- [app/tools/backfill_sw_industry_history_from_tushare.py](/d:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/app/tools/backfill_sw_industry_history_from_tushare.py)

For `DECISION_SUPPORT`, prefer Shenwan Level-1:

```powershell
set TUSHARE_TOKEN=YOUR_TOKEN
python -m app.tools.backfill_sw_industry_history_from_tushare ^
  --start 2010-01-04 ^
  --end 2016-12-31 ^
  --src SW2021 ^
  --level L1 ^
  --master-source TUSHARE_SW2021_L1 ^
  --member-source tushare_sw_l1 ^
  --map-chunk-years 1
```

Notes:

- do not use `--keep-existing-member-source` for disaster recovery
- use `--level L1`, not default L3 logic
- widen `--end` if you want to restore beyond `2016-12-31`

This step writes:

- `cn_board_industry_master`
- `cn_board_industry_member_hist`

And also rebuilds:

- `cn_board_member_map_d`

## Step 2: Rebuild Daily Board Member Map

If you want to rerun map expansion explicitly after Step 1:

```sql
CALL sp_build_board_member_map('2010-01-04', '2016-12-31');
```

Or use the Python wrapper:

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

## Step 3: Ensure Rotation Core DDL Exists

Apply these if the restored DB may be missing SPs or views:

```sql
USE cn_market;

SOURCE docs/DDL/cn_market.sp_refresh_sector_eod_hist.sql;
SOURCE docs/DDL/cn_market.cn_sector_rotation_transition_v.sql;
SOURCE docs/DDL/cn_market.sp_build_sector_rotation_ranked_by_date.sql;
SOURCE docs/DDL/cn_market.sp_build_sector_rotation_signal_by_date.sql;
SOURCE docs/DDL/cn_market.sp_backfill_rot_bt_from_price.sql;
SOURCE docs/DDL/cn_market.sp_repair_rot_bt_nav.sql;
SOURCE docs/DDL/cn_market.sp_refresh_rotation_snap_all.sql;
SOURCE docs/DDL/cn_market.sp_rotation_daily_refresh.sql;
```

## Step 4: Rebuild Rotation Three-Table History

Preferred tool:

- [app/tools/rebuild_rotation_three_tables_from_map.py](/d:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/app/tools/rebuild_rotation_three_tables_from_map.py)

Recommended command:

```powershell
python -m app.tools.rebuild_rotation_three_tables_from_map ^
  --start 2010-01-04 ^
  --end 2016-12-31 ^
  --months-per-chunk 1 ^
  --clear-first 1 ^
  --rank-signal-mode hybrid_base ^
  --retries 8 ^
  --retry-sleep-sec 3
```

This rebuilds:

- `cn_sector_eod_hist_t`
- `cn_sector_rotation_ranked_t`
- `cn_sector_rotation_signal_t`

Recommended mode:

- `hybrid_base`

Reason:

- it is the best fit in this repo for history rebuild from `cn_board_member_map_d`
- it avoids relying only on a daily latest-style workflow

## Step 5: Optional Rotation Backtest Rebuild

If the report or related dashboards use rotation backtest output:

```sql
CALL SP_BACKFILL_ROT_BT_FROM_PRICE('SR_LIVE_DEFAULT', '2016-12-31', 1);
```

This rebuilds:

- `cn_sector_rot_bt_daily_t`

If NAV continuity looks broken after rebuild:

```sql
CALL sp_repair_rot_bt_nav('SR_LIVE_DEFAULT');
```

## Step 6: Optional Snapshot Refresh

If you need the final-day operational snapshot tables:

```sql
CALL SP_REFRESH_ROTATION_SNAP_ALL('SR_LIVE_DEFAULT', '2016-12-31', 1);
```

This refreshes:

- `cn_rotation_entry_snap_t`
- `cn_rotation_holding_snap_t`
- `cn_rotation_exit_snap_t`

Note:

- this is a point-in-time snapshot refresh, not a full-history rebuild

## SQL-Only Recovery Path

If you prefer pure MySQL client execution for rotation history:

```sql
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
```

Then run by year:

```sql
CALL sp_rebuild_rotation_year('2010-01-04', '2010-12-31', 0.30, 0.60);
CALL sp_rebuild_rotation_year('2011-01-01', '2011-12-31', 0.30, 0.60);
CALL sp_rebuild_rotation_year('2012-01-01', '2012-12-31', 0.30, 0.60);
CALL sp_rebuild_rotation_year('2013-01-01', '2013-12-31', 0.30, 0.60);
CALL sp_rebuild_rotation_year('2014-01-01', '2014-12-31', 0.30, 0.60);
CALL sp_rebuild_rotation_year('2015-01-01', '2015-12-31', 0.30, 0.60);
CALL sp_rebuild_rotation_year('2016-01-01', '2016-12-31', 0.30, 0.60);
```

## Validation Checklist

### A. Industry member history

```sql
SELECT
    COUNT(*) AS hist_rows,
    COUNT(DISTINCT board_id) AS board_cnt,
    COUNT(DISTINCT symbol) AS symbol_cnt,
    MIN(valid_from) AS min_valid_from,
    MAX(COALESCE(valid_to, DATE('9999-12-31'))) AS max_valid_to
FROM cn_board_industry_member_hist
WHERE board_id LIKE '801%.SI';
```

```sql
SELECT
    source,
    COUNT(*) AS row_cnt,
    COUNT(DISTINCT board_id) AS board_cnt
FROM cn_board_industry_member_hist
WHERE board_id LIKE '801%.SI'
GROUP BY source
ORDER BY row_cnt DESC;
```

### B. Daily map coverage

```sql
SELECT
    MIN(trade_date) AS min_d,
    MAX(trade_date) AS max_d,
    COUNT(DISTINCT trade_date) AS d_cnt,
    COUNT(*) AS row_cnt
FROM cn_board_member_map_d
WHERE trade_date BETWEEN '2010-01-04' AND '2016-12-31'
  AND sector_type = 'INDUSTRY';
```

```sql
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
```

### C. Rotation three-table coverage

```sql
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
```

```sql
SELECT
    action,
    COUNT(*) AS row_cnt,
    COUNT(DISTINCT signal_date) AS d_cnt
FROM cn_sector_rotation_signal_t
WHERE signal_date BETWEEN '2010-01-04' AND '2016-12-31'
GROUP BY action
ORDER BY row_cnt DESC;
```

### D. Backtest / snapshot validation

```sql
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
```

## Failure Diagnosis Guide

### Case 1: `cn_board_industry_member_hist` still tiny

Likely causes:

- wrong taxonomy level
- loaded L3 instead of L1
- old bad source rows were kept

Action:

- rerun Tushare history backfill with `--level L1`
- do not use `--keep-existing-member-source`

### Case 2: history looks fine but `cn_board_member_map_d` is sparse

Likely causes:

- map rebuild not run
- price calendar coverage insufficient

Action:

- rerun `sp_build_board_member_map`
- verify `cn_stock_daily_price` date coverage

### Case 3: map looks fine but rotation three tables are sparse

Likely causes:

- rotation core SP/view missing
- `sp_refresh_sector_eod_hist` not run
- ranked/signal daily replay not run

Action:

- reapply DDL in the rotation core order
- rerun `rebuild_rotation_three_tables_from_map.py`

### Case 4: signal exists but backtest table is empty or tiny

Likely causes:

- `SP_BACKFILL_ROT_BT_FROM_PRICE` not run
- wrong `run_id`

Action:

- rerun `SP_BACKFILL_ROT_BT_FROM_PRICE('SR_LIVE_DEFAULT', ..., 1)`

### Case 5: all base tables look fine but report still differs

Likely causes:

- report still reads the wrong taxonomy
- report uses latest snapshot-like source instead of `cn_board_member_map_d`
- report-side cache or intermediate materialization not refreshed

Action:

- confirm report SQL reads map-driven historical tables
- confirm sector ids are `801%.SI`
- refresh report-side cache/materialized layer

## Recommended Operations After Recovery

### Daily

Keep these updated:

- `cn_stock_daily_price`
- `cn_index_daily_price`
- `cn_stock_daily_basic`
- `cn_sw_industry_daily`
- rotation daily chain

Recommended:

```powershell
python runner.py --tasks stock,index,stock_basic,sw_industry --asof latest
python runner.py --tasks rotation --asof latest
```

### Weekly

Use weekly as the primary correction point for board member history.

Recommended:

```powershell
python runner.py --tasks board --asof latest
```

For wider SW L1 correction windows:

```powershell
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
```

### Monthly

Run a wider correction window for the mapping and rotation history:

```powershell
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
```

```powershell
python -m app.tools.rebuild_rotation_three_tables_from_map ^
  --start 2024-01-01 ^
  --end 2026-05-18 ^
  --months-per-chunk 1 ^
  --clear-first 1 ^
  --rank-signal-mode hybrid_base ^
  --retries 8 ^
  --retry-sleep-sec 3
```

## Minimal Standing Monitoring

Run these regularly:

```sql
SELECT COUNT(*) AS hist_rows, COUNT(DISTINCT board_id) AS board_cnt
FROM cn_board_industry_member_hist
WHERE board_id LIKE '801%.SI';
```

```sql
SELECT YEAR(trade_date) AS yr, COUNT(DISTINCT trade_date) AS d_cnt
FROM cn_board_member_map_d
WHERE sector_type = 'INDUSTRY'
  AND sector_id LIKE '801%.SI'
GROUP BY YEAR(trade_date)
ORDER BY yr;
```

```sql
SELECT COUNT(DISTINCT trade_date) AS ranked_days
FROM cn_sector_rotation_ranked_t
WHERE trade_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY);
```

```sql
SELECT action, COUNT(*) AS row_cnt
FROM cn_sector_rotation_signal_t
WHERE signal_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
GROUP BY action;
```

## Summary

The stable maintenance order is:

1. `cn_board_industry_member_hist`
2. `cn_board_member_map_d`
3. `cn_sector_eod_hist_t`
4. `cn_sector_rotation_ranked_t`
5. `cn_sector_rotation_signal_t`
6. optional backtest and snapshots
7. `DECISION_SUPPORT`

When results diverge, always debug from the mapping layer upward rather than starting from the final report.
