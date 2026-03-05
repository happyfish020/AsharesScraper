# Rotation Tables Quick Guide

## Purpose
This guide answers one question directly:
- Backtest uses which table?
- Daily reporting/audit uses which table?

Use one filter key consistently in all queries:
- `run_id`
- `trade_date`

---

## Backtest Tables

### 1) Primary backtest timeline (must-have)
- Table: `cn_sector_rot_bt_daily_t`
- Use for:
  - NAV timeline
  - daily return / turnover / cost
  - strategy-level performance replay

### 2) Position-state facts (supporting)
- Table: `cn_sector_rot_pos_daily_t`
- Use for:
  - which sectors are held on each day
  - hold days / min hold / exit flag
  - explain why exits are (or are not) triggered

### 3) Sector state history for backtest attribution (must-have for bear/bull/range)
- Table: `cn_sector_rotation_ranked_t`
- Use for:
  - historical sector state (`state`) per `trade_date`
  - mapping daily backtest performance to regime labels (bear/bull/range)

---

## Daily Reporting / Audit Tables

### 0) Daily sector state source (for bear/bull/range)
- Table: `cn_sector_rotation_ranked_t`
- Use for:
  - today's full market sector regime map (`state`, `tier`, `score`)

### 1) Entry snapshot
- Table: `cn_rotation_entry_snap_t`
- Use for:
  - today's entry pool / rank / suggested weight

### 2) Holding snapshot
- Table: `cn_rotation_holding_snap_t`
- Use for:
  - today's holdings and exit readiness

### 3) Exit snapshot
- Table: `cn_rotation_exit_snap_t`
- Use for:
  - today's exits and execution status

These three snapshot tables are the daily "read model" for dashboards and audit checks.

---

## Signal / Explainability Source

- Table: `cn_sector_rotation_signal_t`
- Use for:
  - verifying ENTER/EXIT/WATCH distribution
  - daily state + transition checks (`state`, `transition`)
  - tracing why snapshot output is empty or summary-only

---

## Minimal Query Templates

### Backtest (NAV curve)
```sql
SELECT trade_date, nav, net_ret, turnover, cost, n_pos
FROM cn_sector_rot_bt_daily_t
WHERE run_id = :RUN_ID
ORDER BY trade_date;
```

### Daily sector state (bear/bull/range)
```sql
SELECT trade_date, sector_type, sector_id, sector_name, state, tier, score
FROM cn_sector_rotation_ranked_t
WHERE trade_date = :DT
ORDER BY sector_type, score DESC;
```

### Daily snapshot counts (audit)
```sql
SELECT 'ENTRY' AS snap, COUNT(*) AS n
FROM cn_rotation_entry_snap_t
WHERE run_id = :RUN_ID AND trade_date = :DT
UNION ALL
SELECT 'HOLDING', COUNT(*)
FROM cn_rotation_holding_snap_t
WHERE run_id = :RUN_ID AND trade_date = :DT
UNION ALL
SELECT 'EXIT', COUNT(*)
FROM cn_rotation_exit_snap_t
WHERE run_id = :RUN_ID AND trade_date = :DT;
```

### Daily signal distribution (sanity)
```sql
SELECT action, COUNT(*) AS n
FROM cn_sector_rotation_signal_t
WHERE signal_date = :DT
GROUP BY action
ORDER BY action;
```

---

## One-line Decision

- For sector regime (bear/bull/range): use `cn_sector_rotation_ranked_t` (daily and history).
- For backtest performance: use `cn_sector_rot_bt_daily_t` (join to ranked by date when you need state attribution).
- For daily report/audit: use `cn_rotation_entry_snap_t`, `cn_rotation_holding_snap_t`, `cn_rotation_exit_snap_t`.

---

## Daily Incremental Operations (Current)

### Runtime Path (already implemented)
- Runner task: `rotation` (`app/tasks/rotation_sector_snapshot_task.py`)
- Default SQL call:
  - `CALL cn_market.SP_ROTATION_DAILY_REFRESH(:p_run_id, :p_trade_date, :p_force, :p_refresh_energy)`
- Trade date semantics:
  - `p_trade_date` uses runner `end_date` (as-of day).
- Current daily-increment tables covered by this call:
  - `cn_sector_eod_hist_t` (index/sector-level daily facts)
  - `cn_sector_rotation_ranked_t` (daily ranked state)
  - `cn_sector_rotation_signal_t` (daily signal)

### Daily operation policy
- Keep one fixed production `run_id` (do not change per day).
- Run order per day:
  1. `stock` task (load latest `cn_stock_daily_price`)
  2. `rotation` task (call `SP_ROTATION_DAILY_REFRESH`)
- Keep parameters:
  - `p_force=0` for daily idempotent run
  - `p_refresh_energy=1` to auto-refresh energy dependency in the same chain
- Only use `p_force=1` for replay/hotfix of a specific date.

### Recommended scheduler command
```bash
python runner.py --tasks stock,rotation --asof latest
```

### Optional env overrides
- `ROTATION_SNAPSHOT_SQL`: override SP call SQL
- `ROTATION_ENERGY_SQL`: deprecated legacy pre-step; default is disabled
- `ROTATION_SNAPSHOT_RUN_ID`: explicit run id

### Daily acceptance checks (4 SQLs)
```sql
-- 1) Today from price
SELECT MAX(trade_date) AS dt FROM cn_stock_daily_price;

-- 2) Signal distribution
SELECT action, COUNT(*) AS n
FROM cn_sector_rotation_signal_t
WHERE signal_date = :DT
GROUP BY action
ORDER BY action;

-- 3) BT daily row exists
SELECT COUNT(*) AS n
FROM cn_sector_rot_bt_daily_t
WHERE run_id = :RUN_ID AND trade_date = :DT;

-- 4) Snapshot rows exist
SELECT 'ENTRY' AS snap, COUNT(*) AS n
FROM cn_rotation_entry_snap_t
WHERE run_id = :RUN_ID AND trade_date = :DT
UNION ALL
SELECT 'HOLDING', COUNT(*)
FROM cn_rotation_holding_snap_t
WHERE run_id = :RUN_ID AND trade_date = :DT
UNION ALL
SELECT 'EXIT', COUNT(*)
FROM cn_rotation_exit_snap_t
WHERE run_id = :RUN_ID AND trade_date = :DT;
```

### Optional check for index-level daily increment
```sql
SELECT COUNT(*) AS n
FROM cn_sector_eod_hist_t
WHERE trade_date = :DT;
```

---

## 3 Tables Playbook (History Backfill + Daily Ops)

This section focuses on exactly these 3 tables:
- `cn_sector_eod_hist_t`
- `cn_sector_rotation_ranked_t`
- `cn_sector_rotation_signal_t`

### A) Historical backfill for 3 tables

#### A0. Critical history mapping constraint (must read)
For historical backfill, do **not** use `cn_board_industry_cons` / `cn_board_concept_cons` as symbol-to-sector source.

Reason:
- `cn_board_*_cons` in this environment is a recent snapshot table (for example, only around `2026-01-09`).
- It does not contain full historical `ASOF_DATE` coverage for old years (such as `2000-2005`).
- Using `*_cons` for old dates will produce wrong or empty mapping.

Required source for history:
- Use `cn_board_member_map_d(trade_date, sector_type, sector_id, symbol)` only.
- If `cn_board_member_map_d` has date gaps, rebuild it from `cn_board_*_member_hist` first.

Pre-flight checks before historical run:
```sql
-- 1) Verify map coverage for the target range
SELECT MIN(trade_date) AS min_d, MAX(trade_date) AS max_d, COUNT(DISTINCT trade_date) AS d_cnt
FROM cn_board_member_map_d
WHERE trade_date BETWEEN :D1 AND :D2;

-- 2) Check if cons tables are snapshot-like (usually near latest date only)
SELECT 'industry_cons' AS src, MIN(asof_date) AS min_d, MAX(asof_date) AS max_d, COUNT(DISTINCT asof_date) AS d_cnt
FROM cn_board_industry_cons
UNION ALL
SELECT 'concept_cons', MIN(asof_date), MAX(asof_date), COUNT(DISTINCT asof_date)
FROM cn_board_concept_cons;
```

Recommended sequence:
1. Backfill `cn_sector_eod_hist_t` by date range (monthly driver).
2. Rebuild `ranked/signal` for each trade day in the same range.

#### A1. Backfill `cn_sector_eod_hist_t` (history)
```sql
CALL sp_backfill_sector_eod_hist_monthly('2025-01-01', '2026-02-27', 0.30, 0.60);
```

#### A2. Backfill `ranked/signal` (history) day by day
Use this when you need to replay old days exactly:
```sql
-- Example single day
CALL SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE('2026-02-26');
CALL SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE('2026-02-26');
```

For a range, execute daily in scheduler/script:
- `SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE(:DT)`
- `SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE(:DT)`

#### A3. History validation for 3 tables
```sql
-- eod_hist coverage
SELECT COUNT(DISTINCT trade_date) AS d_cnt
FROM cn_sector_eod_hist_t
WHERE trade_date BETWEEN :D1 AND :D2;

-- ranked coverage
SELECT COUNT(DISTINCT trade_date) AS d_cnt
FROM cn_sector_rotation_ranked_t
WHERE trade_date BETWEEN :D1 AND :D2;

-- signal coverage
SELECT COUNT(DISTINCT signal_date) AS d_cnt
FROM cn_sector_rotation_signal_t
WHERE signal_date BETWEEN :D1 AND :D2;
```

### B) Daily incremental operations for 3 tables

Daily run uses one orchestrator call:
```sql
CALL SP_ROTATION_DAILY_REFRESH(:RUN_ID, :DT, 0, 1);
```

What this daily call now guarantees:
- Incremental refresh of `cn_sector_eod_hist_t` for `:DT`
- Incremental upsert of `cn_sector_rotation_ranked_t` for `:DT`
- Incremental upsert of `cn_sector_rotation_signal_t` for `:DT`

Runner command (recommended):
```bash
python runner.py --tasks stock,rotation --asof latest
```

Daily acceptance checks:
```sql
SELECT COUNT(*) AS eod_hist_rows
FROM cn_sector_eod_hist_t
WHERE trade_date = :DT;

SELECT COUNT(*) AS ranked_rows
FROM cn_sector_rotation_ranked_t
WHERE trade_date = :DT;

SELECT COUNT(*) AS signal_rows
FROM cn_sector_rotation_signal_t
WHERE signal_date = :DT;
```
