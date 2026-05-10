# P0 Data Audit & DDL Design — GrowthAlpha V8

> **Database**: `cn_market_red` (logical name `cn_market`)
> **Target**: Mainline Strength Engine / Market Breadth Engine / Narrative & Context Layer

---

## 1. Overview

This document describes the data asset audit system and schema migration for the GrowthAlpha V8 `cn_market_red` database. The work is split into two deliverables:

| Deliverable | File | Purpose |
|---|---|---|
| Audit Script | [`scripts/audit_cn_market_data_assets.py`](../scripts/audit_cn_market_data_assets.py) | Audits 20+ tables across 5 tiers |
| Migration Script | [`scripts/apply_ga_p0_schema_migration.py`](../scripts/apply_ga_p0_schema_migration.py) | Idempotent column/index migration |
| DDL SQL | [`sql/ddl/ga_p0_mainline_context_schema.sql`](../sql/ddl/ga_p0_mainline_context_schema.sql) | DDL for `cn_ga_market_context_daily` + indexes |
| This Document | [`docs/P0_DATA_AUDIT_AND_DDL_DESIGN.md`](P0_DATA_AUDIT_AND_DDL_DESIGN.md) | Documentation & report template |

---

## 2. Audit Script

### 2.1 Usage

```bash
python scripts/audit_cn_market_data_assets.py ^
  --db-host 127.0.0.1 ^
  --db-port 3306 ^
  --db-user root ^
  --db-password YOUR_PASSWORD ^
  --db-name cn_market_red ^
  --output-dir reports/data_audit ^
  --write-db ^
  --fail-on-critical
```

### 2.2 Arguments

| Argument | Default | Description |
|---|---|---|
| `--db-host` | `127.0.0.1` | MySQL host |
| `--db-port` | `3306` | MySQL port |
| `--db-user` | `root` | MySQL user |
| `--db-password` | (required) | MySQL password |
| `--db-name` | `cn_market_red` | Database name |
| `--output-dir` | `reports/data_audit` | Output directory |
| `--as-of-date` | today | Reference date for freshness |
| `--write-db` | flag | Write to `cn_ga_data_readiness_daily` |
| `--fail-on-critical` | flag | Exit code 1 if P0 fails |

### 2.3 Audited Tables

#### P0 — Critical (Core Market Data)

| Table | Expected Columns | Freshness Threshold |
|---|---|---|
| `cn_stock_daily_price` | SYMBOL, TRADE_DATE, OPEN, HIGH, LOW, CLOSE, VOLUME, AMOUNT, CHG_PCT | 5 days |
| `cn_index_daily_price` | INDEX_CODE, TRADE_DATE, OPEN, CLOSE, HIGH, LOW, VOLUME, AMOUNT, PRE_CLOSE, CHG_PCT | 5 days |
| `cn_sw_industry_daily` | ts_code, name, trade_date, close, amount | 5 days |
| `cn_stock_monthly_basic` | symbol, trade_date, month_key, total_mv, circ_mv, pe_ttm, pb | 45 days |
| `cn_stock_fina_indicator` | symbol, end_date, ann_date, report_type, netprofit_yoy, q_profit_yoy, or_yoy, eps, roe, grossprofit_margin | 90 days |

#### P1 — Important (Derived / Quality)

| Table | Expected Columns | Freshness Threshold |
|---|---|---|
| `cn_stock_daily_basic` | symbol, trade_date, pe_ttm, pb, total_mv, circ_mv, volume_ratio | 5 days |
| `cn_stock_leader_score_daily` | trade_date, symbol, leader_score, leader_bucket | 5 days |
| `cn_stock_leader_sw_l1_latest_snap` | trade_date, symbol, leader_score, sw_l1_id | 5 days |
| `cn_stock_universe_status_t` | symbol, is_active, last_trade_date | N/A |

#### P2 — Structure (Mapping / Proxy)

| Table | Expected Columns | Freshness Threshold |
|---|---|---|
| `cn_board_member_map_d` | trade_date, sector_type, sector_id, symbol | 7 days |
| `cn_local_industry_map_hist` | symbol, industry_id, in_date, out_date, is_current | N/A |
| `cn_local_industry_proxy_daily` | industry_id, trade_date, member_count, ret_eqw, amount_total | 5 days |

#### P3 — Reporting (Financial Events)

| Table | Expected Columns | Freshness Threshold |
|---|---|---|
| `cn_stock_income` | symbol, end_date, ann_date, report_type, total_revenue, n_income_attr_p | N/A |
| `cn_stock_balancesheet` | symbol, end_date, ann_date, report_type, total_assets, total_liab | N/A |
| `cn_event_disclosure_date` | symbol, end_date, pre_date, actual_date | N/A |
| `cn_event_earnings_forecast` | symbol, ann_date, end_date, forecast_type, p_change_min, p_change_max | N/A |

#### GA Layer (GrowthAlpha Engine)

| Table | Expected Columns | Freshness Threshold |
|---|---|---|
| `cn_ga_mainline_radar_daily` | trade_date, mainline_id, mainline_name, member_count, leader_count, mainline_score, mainline_state, rank_no, reason | 5 days |
| `cn_ga_market_pulse_daily` | trade_date, market_score, market_state, target_exposure, breadth_up_ratio, risk_flag, reason | 5 days |
| `cn_ga_stock_role_map_daily` | trade_date, symbol, stock_name, mainline_id, mainline_name, leader_score, stock_role, role_score, role_reason | 5 days |
| `cn_ga_data_readiness_daily` | trade_date, table_name, status, severity, row_count, max_trade_date, null_rate_summary | N/A |

### 2.4 Status Determination

| Status | Meaning | P0 Severity | P1 Severity | P2/P3/GA Severity |
|---|---|---|---|---|
| OK | Table healthy | INFO | INFO | INFO |
| WARN | High null rates | CRITICAL | WARN | WARN |
| STALE | Data too old | CRITICAL | WARN | WARN |
| MISSING_COLUMNS | Key columns absent | CRITICAL | WARN | WARN |
| MISSING_TABLE | Table doesn't exist | CRITICAL | WARN | WARN |
| EMPTY | Zero rows | CRITICAL | WARN | WARN |

### 2.5 Output Files

1. **CSV**: `reports/data_audit/data_asset_audit_<YYYYMMDD_HHMMSS>.csv`
2. **Markdown**: `reports/data_audit/data_asset_audit_<YYYYMMDD_HHMMSS>.md`
3. **Latest**: `reports/data_audit/data_asset_audit_latest.md`

### 2.6 Readiness Assessment

The report answers three questions:

1. **Ready for Mainline Strength Engine?**
   - Requires: `cn_stock_daily_price` OK, `cn_board_member_map_d` OK, `cn_local_industry_map_hist` OK, `cn_local_industry_proxy_daily` OK, `cn_ga_mainline_radar_daily` OK

2. **Ready for Market Breadth Engine?**
   - Requires: `cn_stock_daily_price` OK, `cn_index_daily_price` OK, `cn_sw_industry_daily` OK, `cn_ga_market_pulse_daily` OK

3. **Ready for Narrative/Context Layer?**
   - Requires: `cn_ga_mainline_radar_daily` OK, `cn_ga_market_pulse_daily` OK, `cn_ga_stock_role_map_daily` OK, `cn_stock_fina_indicator` OK

---

## 3. DDL Schema

### 3.1 New Table: `cn_ga_market_context_daily`

```sql
CREATE TABLE IF NOT EXISTS `cn_ga_market_context_daily` (
    `trade_date`              DATE          NOT NULL,
    `market_regime`           VARCHAR(64)   DEFAULT NULL,
    `mainline_phase`          VARCHAR(64)   DEFAULT NULL,
    `rotation_state`          VARCHAR(64)   DEFAULT NULL,
    `trend_confidence_score`  DECIMAL(10,4) DEFAULT NULL,
    `risk_context`            VARCHAR(128)  DEFAULT NULL,
    `narrative_summary`       TEXT          DEFAULT NULL,
    `data_quality_status`     VARCHAR(32)   DEFAULT NULL,
    `created_at`              TIMESTAMP     NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at`              TIMESTAMP     NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`trade_date`),
    KEY `idx_ga_market_context_daily_regime`   (`market_regime`),
    KEY `idx_ga_market_context_daily_phase`    (`mainline_phase`),
    KEY `idx_ga_market_context_daily_rotation` (`rotation_state`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 3.2 Column Additions to `cn_ga_mainline_radar_daily`

| Column | Type | Purpose |
|---|---|---|
| `rs_60d` | DECIMAL(10,4) | 60-day relative strength |
| `rs_120d` | DECIMAL(10,4) | 120-day relative strength |
| `trend_alignment_score` | DECIMAL(10,4) | Multi-timeframe trend alignment |
| `rotation_rank` | INT | Rank within rotation universe |
| `heat_percentile_5d` | DECIMAL(10,4) | 5-day heat percentile |
| `breakout_ratio` | DECIMAL(10,4) | Ratio of stocks in breakout |
| `new_high_ratio` | DECIMAL(10,4) | Ratio of stocks at new highs |
| `strong_stock_count` | INT | Count of strong stocks |
| `leader_density` | DECIMAL(10,4) | Leader concentration |
| `mainline_phase` | VARCHAR(32) | Lifecycle phase |
| `mainline_confidence` | DECIMAL(10,4) | Confidence score |

### 3.3 Column Additions to `cn_ga_market_pulse_daily`

| Column | Type | Purpose |
|---|---|---|
| `bullish_industry_ratio` | DECIMAL(10,4) | Ratio of bullish industries |
| `neutral_industry_ratio` | DECIMAL(10,4) | Ratio of neutral industries |
| `bearish_industry_ratio` | DECIMAL(10,4) | Ratio of bearish industries |
| `rotation_speed` | DECIMAL(10,4) | Speed of sector rotation |
| `mainline_stability` | DECIMAL(10,4) | Mainline persistence |
| `trend_alignment_avg` | DECIMAL(10,4) | Average trend alignment |
| `industry_expansion_breadth` | DECIMAL(10,4) | Industry expansion breadth |
| `top_mainline_count` | INT | Number of top mainlines |
| `market_phase` | VARCHAR(32) | Market phase classification |

### 3.4 Column Additions to `cn_ga_stock_role_map_daily`

| Column | Type | Purpose |
|---|---|---|
| `breakout_strength` | DECIMAL(10,4) | Breakout strength score |
| `new_high_flag` | TINYINT(1) | Whether stock hit new high |
| `trend_structure_score` | DECIMAL(10,4) | Trend structure quality |
| `volume_expansion_score` | DECIMAL(10,4) | Volume expansion score |
| `role_lifecycle_state` | VARCHAR(32) | Role lifecycle state |
| `candidate_action` | VARCHAR(64) | Suggested action |

---

## 4. Migration Script

### 4.1 Usage

```bash
python scripts/apply_ga_p0_schema_migration.py ^
  --db-host 127.0.0.1 ^
  --db-port 3306 ^
  --db-user root ^
  --db-password YOUR_PASSWORD ^
  --db-name cn_market_red
```

### 4.2 Idempotency Design

The migration script is safe to run multiple times:

1. **Column existence check**: Queries `INFORMATION_SCHEMA.COLUMNS` before each `ALTER TABLE ADD COLUMN`
2. **Table existence check**: Uses `CREATE TABLE IF NOT EXISTS` for `cn_ga_market_context_daily`
3. **Index existence check**: Queries `INFORMATION_SCHEMA.STATISTICS` before each `CREATE INDEX`
4. **No destructive operations**: Never drops tables, columns, or data

### 4.3 Migration Steps

| Step | Action |
|---|---|
| 1 | Add columns to `cn_ga_mainline_radar_daily` |
| 2 | Add columns to `cn_ga_market_pulse_daily` |
| 3 | Add columns to `cn_ga_stock_role_map_daily` |
| 4 | Create `cn_ga_market_context_daily` if not exists |
| 5 | Create indexes on all GA-layer tables |

---

## 5. Syntax Verification

Both Python scripts pass `py_compile`:

```bash
python -m py_compile scripts/audit_cn_market_data_assets.py   # PASS
python -m py_compile scripts/apply_ga_p0_schema_migration.py  # PASS
```

---

## 6. Audit Report Template

When the audit script runs, it generates a markdown report with the following sections:

### 6.1 Executive Summary

```
| Metric | Value |
|--------|-------|
| Total Tables | 20 |
| Healthy (OK) | ... |
| Warning (WARN/STALE/MISSING_COLUMNS) | ... |
| Failed (FAIL/MISSING_TABLE/EMPTY) | ... |
| P0 Critical Failures | ... |
```

### 6.2 Critical Table Status

Per-table status for P0 and P1 tables with row counts, max dates, and lag days.

### 6.3 Per-Tier Detail

Full detail for all 5 tiers with missing columns and null rates.

### 6.4 Date Coverage Summary

Min/max dates and distinct trading days per table.

### 6.5 Missing Columns

Tables where expected key columns are absent.

### 6.6 Stale Tables

Tables exceeding their freshness threshold.

### 6.7 Null Rate Analysis

Key columns with >20% null rates.

### 6.8 Recommended Next Actions

Prioritized action items for non-OK tables.

### 6.9 Readiness Assessment

Three boolean assessments:
- Mainline Strength Engine: READY / NOT READY
- Market Breadth Engine: READY / NOT READY
- Narrative / Context Layer: READY / NOT READY

---

## 7. File Inventory

```
scripts/
  audit_cn_market_data_assets.py       # Main audit script (20+ tables, 5 tiers)
  apply_ga_p0_schema_migration.py      # Idempotent schema migration

sql/ddl/
  ga_p0_mainline_context_schema.sql    # DDL for cn_ga_market_context_daily + indexes

docs/
  P0_DATA_AUDIT_AND_DDL_DESIGN.md     # This document
```

---

## 8. Quick Reference

### Run Audit

```bash
python scripts/audit_cn_market_data_assets.py ^
  --db-host 127.0.0.1 --db-port 3306 --db-user root ^
  --db-password YOUR_PASSWORD --db-name cn_market_red ^
  --output-dir reports/data_audit --write-db --fail-on-critical
```

### Run Migration

```bash
python scripts/apply_ga_p0_schema_migration.py ^
  --db-host 127.0.0.1 --db-port 3306 --db-user root ^
  --db-password YOUR_PASSWORD --db-name cn_market_red
```

### Verify Syntax

```bash
python -m py_compile scripts/audit_cn_market_data_assets.py
python -m py_compile scripts/apply_ga_p0_schema_migration.py
```
