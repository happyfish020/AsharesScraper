# P3 Unified Alpha Engine — GrowthAlpha V8

## Overview

The **Unified Alpha Engine** is the P3 component of GrowthAlpha V8. It computes a daily unified alpha score for each A-share stock by combining **8 alpha factors** spanning fundamental quality, industry mainline dynamics, capital flow, trend quality, lifecycle positioning, and risk/crowding filters.

The engine produces:
- **`cn_stock_quality_score_daily`** — 5 sub-scores of fundamental quality + composite quality score
- **`cn_unified_alpha_score_daily`** — 8 factor scores + weighted final score + cross-sectional bucket classification + explainability output

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  P3 Unified Alpha Engine                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  cn_stock_fundamental_daily ─┐                              │
│  cn_stock_fina_indicator ─┤                                 │
│  cn_stock_income ─────────┤→ cn_stock_quality_score_daily ─┐│
│  cn_stock_balancesheet ───┘                              │  │
│                                                          │  │
│  cn_mainline_strength_daily ─────────────────────────────┤  │
│  cn_industry_capital_flow_daily ─────────────────────────┤  │
│  cn_ga_mainline_radar_daily ────────────────────────────────┤  │
│  cn_stock_daily_price ───────────────────────────────────┤→│
│  cn_mainline_lifecycle_daily ────────────────────────────┤  │
│  cn_ga_market_pulse_daily ──────────────────────────────────┤  │
│  cn_ga_stock_role_map_daily ────────────────────────────────┘  │
│                                          cn_unified_alpha_score_daily
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Part A: `cn_stock_quality_score_daily`

**Builder**: [`scripts/build_stock_quality_score_daily.py`](../scripts/build_stock_quality_score_daily.py)
**DDL**: [`sql/create_stock_quality_score_daily.sql`](../sql/create_stock_quality_score_daily.sql)

**Input Sources**:
| Source Table | Fields Used |
|---|---|
| `cn_stock_fundamental_daily` | total_revenue, net_profit, operating_profit, operating_cashflow, total_assets, total_liabilities, gross_margin, roe |
| `cn_stock_fina_indicator` | yoyprofit, yoy_tr, roe_weighted, grossprofit_margin, profit_to_gr, debt_to_assets, current_ratio, ocf_to_or |
| `cn_stock_income` | total_revenue, net_profit, operating_profit |
| `cn_stock_balancesheet` | total_assets, total_liabilities, current_assets, current_liabilities |

**5 Sub-Scores** (each [0, 1]):
1. **growth_acceleration_score** — YoY revenue growth + YoY profit growth
2. **cashflow_score** — Operating cash flow / operating revenue ratio
3. **debt_control_score** — Asset-liability ratio (inverse) + current ratio
4. **margin_stability_score** — Gross margin stability (low std dev over 4 quarters)
5. **profitability_score** — ROE + net profit margin

**Composite**: `quality_score = equal-weighted average of 5 sub-scores`

**Risk Flag**: `fundamental_risk_flag` — `NONE` / `HIGH_DEBT` / `NEGATIVE_EARNINGS` / `CASHFLOW_WEAK` / `DATA_INSUFFICIENT`

### Part B: `cn_unified_alpha_score_daily`

**Builder**: [`scripts/build_unified_alpha_score_daily.py`](../scripts/build_unified_alpha_score_daily.py)
**DDL**: [`sql/create_unified_alpha_score_daily.sql`](../sql/create_unified_alpha_score_daily.sql)

**8 Alpha Factors**:

| # | Factor | Weight | Source | Description |
|---|--------|--------|--------|-------------|
| 1 | `quality_score` | 0.10 | `cn_stock_quality_score_daily` | Composite fundamental quality |
| 2 | `growth_acceleration_score` | 0.10 | `cn_stock_quality_score_daily` | Revenue/profit growth acceleration |
| 3 | `mainline_strength_score` | 0.20 | `cn_mainline_strength_daily` | Industry mainline strength |
| 4 | `capital_concentration_score` | 0.15 | `cn_industry_capital_flow_daily` | Capital concentration in industry |
| 5 | `leader_dominance_score` | 0.15 | `cn_ga_mainline_radar_daily` | Leader density + new high ratio + breakout ratio |
| 6 | `trend_quality_score` | 0.15 | `cn_stock_daily_price` | Price trend quality (MA, volume, return) |
| 7 | `lifecycle_position_score` | 0.10 | `cn_mainline_lifecycle_daily` | Lifecycle state mapping |
| 8 | `risk_crowding_score` | 0.05 | `cn_ga_market_pulse_daily` + `cn_ga_mainline_radar_daily` | Market risk + crowding filter |

**Final Score**: `final_score = Σ(weight_i × score_i) / Σ(weight_i)`, clipped to [0, 1]

**Alpha Buckets** (cross-sectional percentile per trade_date):
| Bucket | Percentile Threshold |
|--------|---------------------|
| `TOP_1` | ≥ 99% |
| `TOP_5` | ≥ 95% |
| `TOP_10` | ≥ 90% |
| `TOP_20` | ≥ 80% |
| `WATCH` | ≥ 60% |
| `NEUTRAL` | ≥ 30% |
| `AVOID` | < 30% |

**Explainability Output**:
- `explanation` — Natural language description of stock's alpha classification
- `top_factors` — Top 3 contributing factors (highest scores)
- `weak_factors` — Bottom 3 detracting factors (lowest scores, < 0.5)
- `flags` — Risk flags, lifecycle state, alpha tier

## CLI Usage

### Build cn_stock_quality_score_daily

```bash
# Basic usage
python scripts/build_stock_quality_score_daily.py --start 2026-01-01 --end 2026-03-30

# With replace mode
python scripts/build_stock_quality_score_daily.py --start 2026-01-01 --end 2026-03-30 --replace

# Dry run (no DB writes)
python scripts/build_stock_quality_score_daily.py --start 2026-03-30 --end 2026-03-30 --dry-run --verbose

# Custom database
python scripts/build_stock_quality_score_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red
```

### Build cn_unified_alpha_score_daily

```bash
# Basic usage
python scripts/build_unified_alpha_score_daily.py --start 2026-01-01 --end 2026-03-30

# With replace mode
python scripts/build_unified_alpha_score_daily.py --start 2026-01-01 --end 2026-03-30 --replace

# Dry run (no DB writes)
python scripts/build_unified_alpha_score_daily.py --start 2026-03-30 --end 2026-03-30 --dry-run --verbose

# Custom database
python scripts/build_unified_alpha_score_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red
```

### Validate

```bash
# Validate cn_unified_alpha_score_daily
python scripts/validate_unified_alpha_score_daily.py --start 2026-01-01 --end 2026-03-30

# With verbose output
python scripts/validate_unified_alpha_score_daily.py --start 2026-01-01 --end 2026-03-30 --verbose

# Fail on empty data
python scripts/validate_unified_alpha_score_daily.py --start 2026-01-01 --end 2026-03-30 --min-rows 100 --fail-on-empty
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--start` | `2010-01-01` | Start date (YYYY-MM-DD) |
| `--end` | today | End date (YYYY-MM-DD) |
| `--db-name` | `cn_market_red` | Database name |
| `--db-host` | `127.0.0.1` | MySQL host |
| `--db-port` | `3306` | MySQL port |
| `--db-user` | `root` | MySQL user |
| `--db-password` | env `ASHARE_MYSQL_PASSWORD` | MySQL password |
| `--dry-run` | `False` | Compute but do not write to DB |
| `--replace` | `False` | Delete + re-insert existing rows |
| `--output-dir` | `reports/unified_alpha/` | Override report output directory |
| `--verbose` | `False` | Verbose logging |

## Output

### Reports

Both builders generate reports in the `reports/` directory:

- **`reports/stock_quality/`** — CSV + Markdown reports for stock quality scores
- **`reports/unified_alpha/`** — CSV + Markdown reports for unified alpha scores

Report files are named: `{prefix}_{start}_{end}_{timestamp}.csv` / `.md`

### Database Tables

- **`cn_stock_quality_score_daily`** — Primary key: `(trade_date, symbol)`
- **`cn_unified_alpha_score_daily`** — Primary key: `(trade_date, symbol)`

## Validation Checks

The validation script (`scripts/validate_unified_alpha_score_daily.py`) performs:

1. Table exists
2. Required columns exist (19 columns)
3. Data exists for date range
4. `final_score` in [0, 1]
5. All 8 factor scores in [0, 1]
6. `alpha_bucket` values are valid (7 valid buckets)
7. No duplicate primary keys
8. Explanation NULL ratio ≤ 5%
9. Final score consistent with weighted factor average
10. Bucket distribution summary (verbose mode)

## Dependencies

- Python 3.10+
- pandas, numpy
- SQLAlchemy 2.0+ with pymysql driver
- MySQL 8.0+ database with `cn_market_red` schema

## File Inventory

| File | Description |
|------|-------------|
| `sql/create_stock_quality_score_daily.sql` | DDL for cn_stock_quality_score_daily table |
| `sql/create_unified_alpha_score_daily.sql` | DDL for cn_unified_alpha_score_daily table |
| `scripts/build_stock_quality_score_daily.py` | Builder for stock quality scores |
| `scripts/build_unified_alpha_score_daily.py` | Builder for unified alpha scores |
| `scripts/validate_unified_alpha_score_daily.py` | Validation script |
| `docs/UNIFIED_ALPHA_ENGINE.md` | This documentation |
