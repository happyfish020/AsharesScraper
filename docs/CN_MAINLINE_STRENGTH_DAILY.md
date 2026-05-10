# cn_mainline_strength_daily — 行业主线强度日表

## Overview

`cn_mainline_strength_daily` is a P0 upstream table in the GrowthAlpha V8 pipeline. It computes daily industry-level mainline strength scores by aggregating radar metrics, capital flow data, and leader scores into a single composite strength score with phase classification.

The table fills a critical upstream data gap: the previous `cn_mainline_strength_daily` had a different schema (stock-level, not industry-level) and contained 0 rows for the target date range (2026-01-01 ~ 2026-03-30).

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `trade_date` | date | Trading date (PK) |
| `industry_id` | varchar(32) | Industry ID (SW L1) (PK) |
| `industry_name` | varchar(128) | Industry name |
| `mainline_strength` | decimal(18,8) | Composite mainline strength score [0,1] |
| `trend_alignment_score` | decimal(18,8) | Trend consistency score [0,1] |
| `breakout_ratio` | decimal(18,8) | Breakout pattern ratio [0,1] |
| `new_high_ratio` | decimal(18,8) | New high ratio [0,1] |
| `leader_density` | decimal(18,8) | Leader density [0,1] |
| `leader_count` | int | Number of leader stocks |
| `strong_stock_count` | int | Number of strong stocks |
| `capital_concentration_score` | decimal(18,8) | Capital concentration score [0,1] |
| `rotation_rank` | int | Rotation rank (1=strongest) |
| `mainline_phase` | varchar(32) | Mainline phase classification |
| `created_at` | timestamp | Record creation timestamp |
| `updated_at` | timestamp | Record update timestamp |

**Primary Key:** `(trade_date, industry_id)`

## Core Computation

### mainline_strength (Weighted Composite)

```
mainline_strength = 0.30 * mainline_score          (from cn_ga_mainline_radar_daily)
                   + 0.20 * capital_concentration_score  (from cn_industry_capital_flow_daily)
                   + 0.15 * leader_density          (from cn_ga_mainline_radar_daily)
                   + 0.15 * breakout_ratio          (from cn_ga_mainline_radar_daily)
                   + 0.20 * trend_alignment_score   (from cn_ga_mainline_radar_daily)
```

All sub-scores are clipped to [0, 1].

### mainline_phase Classification

| Phase | Threshold | Description |
|-------|-----------|-------------|
| `DOMINANT` | >= 0.85 | Market-dominating mainline |
| `EXPANDING` | [0.70, 0.85) | Strong expansion phase |
| `EMERGING` | [0.55, 0.70) | Early-stage emergence |
| `DIVERGING` | [0.40, 0.55) with weakening breakout | Divergence detected |
| `DECAYING` | < 0.40 | Declining mainline |
| `UNKNOWN` | N/A | Fallback (no data) |

**DIVERGING detection:** When `mainline_strength` is in [0.40, 0.55) AND `breakout_ratio < trend_alignment_score * 0.8`, the phase is classified as DIVERGING (breakout weakening relative to trend). Otherwise, it defaults to EMERGING.

## Source Tables

| Table | Role | Key Columns |
|-------|------|-------------|
| `cn_ga_mainline_radar_daily` | Primary radar metrics | mainline_score, leader_density, breakout_ratio, new_high_ratio, rotation_rank, trend_alignment_score, leader_count, strong_stock_count |
| `cn_industry_capital_flow_daily` | Capital concentration (optional) | concentration_score, market_share |
| `cn_stock_leader_score_daily` | Per-stock leader scores | leader_score, leader_bucket, industry_id |
| `cn_ga_stock_role_map_daily` | Stock-to-mainline mapping | symbol, mainline_id, stock_role |
| `cn_local_industry_map_hist` | Industry name mapping | industry_id, industry_name |
| `cn_stock_daily_price` | Price data (UPPERCASE columns) | SYMBOL, TRADE_DATE, CLOSE, CHG_PCT |
| `cn_stock_daily_basic` | Market cap data | symbol, trade_date, total_mv, circ_mv |

## Files

| File | Description |
|------|-------------|
| `sql/create_cn_mainline_strength_daily.sql` | DDL for table creation |
| `scripts/build_cn_mainline_strength_daily.py` | Build script (standalone, ~830 lines) |
| `scripts/validate_cn_mainline_strength_daily.py` | Validation script (~430 lines) |

## Usage

### Build

```bash
# Full build with replace
python scripts/build_cn_mainline_strength_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --replace --verbose

# Dry-run (compute only, no DB write)
python scripts/build_cn_mainline_strength_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --dry-run
```

### Validate

```bash
# Full validation
python scripts/validate_cn_mainline_strength_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --min-rows 10 --fail-on-empty

# Skip source table pre-checks
python scripts/validate_cn_mainline_strength_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --skip-source-checks
```

## Validation Checks

1. **Source table existence** — All 7 source tables must exist
2. **Source column existence** — Required columns present in each source
3. **Column case sensitivity** — No case mismatches
4. **Output table exists** — `cn_mainline_strength_daily` created
5. **Required columns exist** — All 13 output columns present
6. **Data exists** — Non-empty for date range
7. **mainline_strength in [0,1]** — Score range integrity
8. **All sub-scores in [0,1]** — 6 score columns in valid range
9. **mainline_phase valid** — Only known phase values
10. **No duplicate PKs** — Unique `(trade_date, industry_id)`
11. **rotation_rank sequential** — Rank 1..N per date
12. **Phase consistency** — Phase matches strength thresholds

## Dependencies

- Python 3.10+
- pandas, numpy, sqlalchemy, pymysql
- MySQL database `cn_market_red`
- Environment variable `ASHARE_MYSQL_PASSWORD` (default: `sec_Bobo123`)
