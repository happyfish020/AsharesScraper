# P0 Mainline Data — Table Reference

This document describes every table involved in the **P0 Mainline Data Foundation** pipeline, including what each table stores, its source, and how it connects to other tables.

---

## 1. Core Source Tables (Raw / External)

### [`cn_stock_daily_price`](docs/DDL/Dump_c_market_20260305.sql:906)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Daily OHLCV price data for all A-share stocks. The foundational price table for all downstream calculations. |
| **Source** | Tushare Pro `daily` API (backfilled via [`sync_cn_stock_daily_price_from_tushare.py`](app/tools/sync_cn_stock_daily_price_from_tushare.py)) |
| **Key Columns** | `SYMBOL`, `TRADE_DATE`, `OPEN`, `CLOSE`, `PRE_CLOSE`, `HIGH`, `LOW`, `VOLUME`, `AMOUNT`, `CHG_PCT`, `TURNOVER_RATE` |
| **PK** | `(SYMBOL, TRADE_DATE)` |
| **Collation** | `utf8mb4_unicode_ci` |
| **Used By** | Every aggregation script — this is the primary price source |

### [`cn_stock_daily_basic`](docs/DDL/cn_market.cn_stock_daily_basic.sql)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Daily fundamental metrics per stock: market cap, turnover rate, PE/PB ratios. |
| **Source** | Tushare Pro `daily_basic` API |
| **Key Columns** | `symbol`, `trade_date`, `total_mv`, `circ_mv`, `turnover_rate_f`, `pe`, `pb` |
| **PK** | `(symbol, trade_date)` |
| **Collation** | `utf8mb4_general_ci` |
| **Used By** | [`build_local_industry_proxy_daily.py`](scripts/build_local_industry_proxy_daily.py) — provides `total_mv`/`circ_mv` for market-cap-weighted calculations |

### [`cn_sw_industry_daily`](docs/DDL/cn_market.cn_sw_industry_daily.sql)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Shenwan (SW) industry index daily行情 (OHLCV + PE/PB) directly from Tushare. This is the **official** SW industry index data. |
| **Source** | Tushare Pro `sw_daily` API (backfilled via [`backfill_sw_industry_daily.py`](scripts/backfill_sw_industry_daily.py)) |
| **Key Columns** | `ts_code`, `trade_date`, `name`, `open`, `close`, `pct_change`, `vol`, `amount`, `pe`, `pb`, `float_mv` |
| **PK** | `(ts_code, trade_date)` |
| **Collation** | `utf8mb4_general_ci` |
| **Used By** | Validation / comparison against locally-computed proxy data |

### [`cn_stock_leader_score_daily`](docs/DDL/cn_market.cn_stock_leader_score_v2.sql)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Per-stock daily leader score (0-3 scale) based on market-cap rank, liquidity, trend, and breakout signals. Identifies which stocks are "leaders" within their industry. |
| **Source** | Computed from views [`cn_stock_leader_score_v1`](docs/DDL/cn_market.cn_stock_leader_score_v1.sql) / [`cn_stock_leader_score_v2`](docs/DDL/cn_market.cn_stock_leader_score_v2.sql) |
| **Key Columns** | `symbol`, `trade_date`, `leader_score`, `leader_structural`, `leader_liquidity`, `leader_trend`, `breakout_strength` |
| **Collation** | `utf8mb4_unicode_ci` |
| **Used By** | [`build_local_industry_proxy_daily.py`](scripts/build_local_industry_proxy_daily.py) — optional input for `leader_return` calculation |

### [`cn_stock_income`](docs/DDL/cn_market.cn_stock_income.sql)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Historical quarterly income statement snapshots from Tushare `income` API. Contains full income statement line items (revenue, operating profit, net income, EPS, etc.) for each reporting period. |
| **Source** | Tushare Pro `income` API |
| **Key Columns** | `symbol`, `end_date`, `ann_date`, `f_ann_date`, `report_type`, `total_revenue`, `revenue`, `n_income_attr_p`, `basic_eps`, `ebit`, `ebitda` |
| **PK** | `(symbol, end_date)` |
| **Collation** | `utf8mb4_general_ci` |
| **Used By** | [`stock_fundamental_daily.py`](data_pipeline/builders/stock_fundamental_daily.py) — source data for `local_stock_income_q` |

### [`cn_stock_balancesheet`](docs/DDL/cn_market.cn_stock_balancesheet.sql)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Historical quarterly balance sheet snapshots from Tushare `balancesheet` API. Contains assets, liabilities, and equity line items. |
| **Source** | Tushare Pro `balancesheet` API |
| **Key Columns** | `symbol`, `end_date`, `ann_date`, `f_ann_date`, `report_type`, `total_assets`, `total_liab`, `fix_assets`, `inventories` |
| **PK** | `(symbol, end_date)` |
| **Collation** | `utf8mb4_general_ci` |
| **Used By** | [`stock_fundamental_daily.py`](data_pipeline/builders/stock_fundamental_daily.py) — source data for `local_stock_balancesheet_q` |

### [`cn_stock_fina_indicator`](docs/DDL/cn_market.cn_stock_fina_indicator.sql)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Historical quarterly financial indicator snapshots from Tushare `fina_indicator` API. Contains derived ratios (ROE, gross margin, YoY growth, debt ratio). |
| **Source** | Tushare Pro `fina_indicator` API |
| **Key Columns** | `symbol`, `end_date`, `ann_date`, `report_type`, `roe`, `or_yoy`, `netprofit_yoy`, `grossprofit_margin`, `debt_to_assets`, `ocfps` |
| **PK** | `(symbol, end_date)` |
| **Collation** | `utf8mb4_general_ci` |
| **Used By** | [`stock_fundamental_daily.py`](data_pipeline/builders/stock_fundamental_daily.py) — source data for `local_stock_fina_indicator_q` |

---

## 2. Industry Master & Membership Tables

### [`cn_local_industry_map_hist`](scripts/build_local_industry_map_hist.py:1)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Historical industry membership mapping: which stock belonged to which industry on which dates. Tracks `in_date` / `out_date` for each membership. |
| **Source** | Built by [`build_local_industry_map_hist.py`](scripts/build_local_industry_map_hist.py) from Tushare `index_member_all` API + existing `cn_board_member_map_d` |
| **Key Columns** | `symbol`, `industry_id`, `industry_name`, `industry_level` (L1/L2/L3), `in_date`, `out_date`, `is_manual_override`, `source` |
| **PK** | `(symbol, industry_id, in_date)` |
| **Collation** | `utf8mb4_0900_ai_ci` |
| **Used By** | [`build_local_industry_proxy_daily.py`](scripts/build_local_industry_proxy_daily.py) — JOINs on `symbol` + date range to determine which stocks belong to each industry on each trade date |

### [`cn_local_industry_master`](data_pipeline/builders/sw_industry_master.py:1)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Industry classification master table: maps `industry_id` to `industry_name`, `industry_level`, `parent_id`, and `src` (e.g., SW2021). |
| **Source** | Built by [`sw_industry_master.py`](data_pipeline/builders/sw_industry_master.py) from Tushare `index_classify` API |
| **Key Columns** | `industry_id`, `industry_name`, `industry_level`, `parent_id`, `src` |
| **PK** | `(industry_id, src)` |
| **Used By** | [`sw_industry_member_hist.py`](data_pipeline/builders/sw_industry_member_hist.py) — provides industry metadata when building membership history |

---

## 3. Financial Quarterly Tables (Interim Layer)

> **Why do `cn_local_stock_*_q` tables exist alongside `cn_stock_*`?**
> The `cn_stock_*` tables (e.g. [`cn_stock_income`](docs/DDL/cn_market.cn_stock_income.sql)) are the **full raw source** tables from Tushare, containing every column the API provides. The `cn_local_stock_*_q` tables are a **curated subset** — they extract only the columns needed by the downstream [`cn_stock_fundamental_daily`](#cn_stock_fundamental_daily) builder, with a consistent schema and a `source` column tracking provenance. They are **not duplicates**; they are an **interim materialization layer** that decouples the pipeline from upstream schema changes.

### [`cn_local_stock_income_q`](docs/DDL/ga_mainline_data_backfill_system.sql:63)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Curated subset of [`cn_stock_income`](#cn_stock_income) — only the columns needed for fundamental scoring: `total_revenue`, `revenue`, `n_income_attr_p`. Populated by [`stock_fundamental_daily.py`](data_pipeline/builders/stock_fundamental_daily.py) via `INSERT ... ON DUPLICATE KEY UPDATE` from `cn_stock_income`. |
| **Source** | Populated from [`cn_stock_income`](#cn_stock_income) |
| **Key Columns** | `symbol`, `end_date`, `ann_date`, `f_ann_date`, `report_type`, `total_revenue`, `revenue`, `n_income_attr_p`, `source` |
| **PK** | `(symbol, end_date)` |
| **Collation** | `utf8mb4_unicode_ci` |
| **Used By** | [`stock_fundamental_daily.py`](data_pipeline/builders/stock_fundamental_daily.py) — joined with `cn_local_stock_balancesheet_q` and `cn_local_stock_fina_indicator_q` to materialize daily fundamental rows |

### [`cn_local_stock_balancesheet_q`](docs/DDL/ga_mainline_data_backfill_system.sql:79)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Curated subset of [`cn_stock_balancesheet`](#cn_stock_balancesheet) — only `inventory`, `fixed_assets`, `total_assets`, `total_liab`. Populated from `cn_stock_balancesheet`. |
| **Source** | Populated from [`cn_stock_balancesheet`](#cn_stock_balancesheet) |
| **Key Columns** | `symbol`, `end_date`, `ann_date`, `f_ann_date`, `report_type`, `inventory`, `fixed_assets`, `total_assets`, `total_liab`, `source` |
| **PK** | `(symbol, end_date)` |
| **Collation** | `utf8mb4_unicode_ci` |
| **Used By** | [`stock_fundamental_daily.py`](data_pipeline/builders/stock_fundamental_daily.py) |

### [`cn_local_stock_fina_indicator_q`](docs/DDL/ga_mainline_data_backfill_system.sql:97)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Curated subset of [`cn_stock_fina_indicator`](#cn_stock_fina_indicator) — only `revenue_yoy`, `profit_yoy`, `roe`, `gross_margin`, `debt_to_assets`, `ocfps`. Populated from `cn_stock_fina_indicator`. |
| **Source** | Populated from [`cn_stock_fina_indicator`](#cn_stock_fina_indicator) |
| **Key Columns** | `symbol`, `end_date`, `ann_date`, `report_type`, `revenue_yoy`, `profit_yoy`, `roe`, `gross_margin`, `debt_to_assets`, `ocfps`, `source` |
| **PK** | `(symbol, end_date)` |
| **Collation** | `utf8mb4_unicode_ci` |
| **Used By** | [`stock_fundamental_daily.py`](data_pipeline/builders/stock_fundamental_daily.py) |

### [`cn_stock_fundamental_daily`](docs/DDL/ga_mainline_data_backfill_system.sql:115)
| Aspect | Description |
|--------|-------------|
| **Purpose** | **Daily materialized view** of the latest available quarterly fundamental data per stock. For each trade date, it carries forward the most recent quarterly report data (using `ann_date` as the effective date). This is the table actually used by downstream strength calculations. |
| **Source** | Built by [`stock_fundamental_daily.py`](data_pipeline/builders/stock_fundamental_daily.py) by joining `cn_local_stock_income_q` + `cn_local_stock_balancesheet_q` + `cn_local_stock_fina_indicator_q` with [`cn_stock_daily_price`](#cn_stock_daily_price) |
| **Key Columns** | `symbol`, `trade_date`, `report_end_date`, `ann_date`, `revenue_yoy`, `profit_yoy`, `roe`, `gross_margin`, `debt_to_assets`, `ocfps`, `inventory`, `fixed_assets`, `source` |
| **PK** | `(symbol, trade_date)` |
| **Collation** | `utf8mb4_unicode_ci` |
| **Used By** | [`cn_mainline_strength_daily.py`](data_pipeline/builders/mainline_strength_daily.py) — provides `earnings_score` via industry aggregation |

---

## 4. Computed Proxy Tables (P0 Mainline)

### [`cn_local_industry_proxy_daily`](scripts/build_local_industry_proxy_daily.py:1)
| Aspect | Description |
|--------|-------------|
| **Purpose** | **Core P0 table.** Daily computed industry-level proxy metrics derived from constituent stock prices. This is the **locally-computed alternative** to `cn_sw_industry_daily` (which comes from Tushare directly). |
| **Source** | Built by [`build_local_industry_proxy_daily.py`](scripts/build_local_industry_proxy_daily.py) |
| **Input Tables** | [`cn_local_industry_map_hist`](#cn_local_industry_map_hist), [`cn_stock_daily_price`](#cn_stock_daily_price), [`cn_stock_daily_basic`](#cn_stock_daily_basic), optionally [`cn_stock_leader_score_daily`](#cn_stock_leader_score_daily) |
| **Key Columns** | |
| | `industry_id` — industry code |
| | `industry_name` — industry name |
| | `trade_date` — trading date |
| | `member_count` — number of constituent stocks in this industry on this date |
| | `ret_eqw` — equal-weighted average return of all constituents |
| | `amount_total` — total trading amount (sum of all constituent amounts) |
| | `turnover_avg` — average turnover rate across constituents |
| | `market_cap_total` — total market cap (sum of all constituent market caps) |
| | `leader_return` — return of the top leader stock (by leader_score or top-5 return avg) |
| | `top5_concentration` — top-5 market cap / total market cap ratio |
| | `industry_level` — L1/L2/L3 |
| | `source` — always `'local_proxy_from_stock_daily'` |
| **PK** | `(industry_id, trade_date)` |
| **Collation** | `utf8mb4_0900_ai_ci` |
| **Used By** | Downstream GA-layer tables (radar, pulse, strength) |

### [`cn_industry_capital_flow_daily`](data_pipeline/builders/industry_capital_flow.py)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Daily industry-level capital flow and relative strength metrics. Tracks leader counts, breakout/trend counts, and relative return vs market. |
| **Source** | Built by [`industry_capital_flow.py`](data_pipeline/builders/industry_capital_flow.py) |
| **Key Columns** | `industry_id`, `trade_date`, `leader_count`, `breakout_count`, `trend_count`, `relative_return`, `market_turnover_ratio` |
| **Used By** | [`cn_mainline_strength_daily.py`](data_pipeline/builders/mainline_strength_daily.py) — provides capital flow and relative return data |

### [`cn_mainline_strength_daily`](data_pipeline/builders/mainline_strength_daily.py)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Daily mainline strength scores and lifecycle states. Combines capital flow, relative return, leader ratio, earnings scores, and expansion scores into a composite `strength_score` (0-100). |
| **Source** | Built by [`mainline_strength_daily.py`](data_pipeline/builders/mainline_strength_daily.py) |
| **Key Columns** | `trade_date`, `mainline_name`, `strength_score`, `leader_count`, `capital_ratio`, `earnings_score`, `trend_days`, `expansion_score`, `lifecycle_state` (CONFIRM/IGNITE/EXPAND/FADE/NEUTRAL) |
| **Used By** | GA-layer radar and pulse tables |

### [`cn_stock_fundamental_daily`](data_pipeline/builders/stock_fundamental_daily.py)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Daily stock-level fundamental scores (revenue growth, profit growth, ROE, gross margin, debt ratio). |
| **Source** | Built by [`stock_fundamental_daily.py`](data_pipeline/builders/stock_fundamental_daily.py) |
| **Used By** | [`cn_mainline_strength_daily.py`](data_pipeline/builders/mainline_strength_daily.py) — provides `earnings_score` via industry aggregation |

---

## 5. GA-Layer Tables (GrowthAlpha Context)

### [`cn_ga_mainline_radar_daily`](sql/ddl/ga_p0_mainline_context_schema.sql:42)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Daily radar snapshot of each mainline (industry). Tracks mainline state, strength trend, leader density, and rotation signals. Used by Factor 5 (leader_dominance_score) and Factor 8 (risk_crowding_score) in the Unified Alpha Engine. |
| **Build Script** | [`scripts/build_ga_mainline_radar_daily.py`](scripts/build_ga_mainline_radar_daily.py) |
| **Sources** | `cn_ga_stock_role_map_daily`, `cn_stock_daily_price`, `cn_stock_mainline_strength_daily` |
| **Key Alpha Columns** | `leader_density` (leader_count/member_count), `new_high_ratio` (rs_pct ≥ 0.85 proxy), `breakout_ratio` (from mainline_strength), `trend_alignment_score` (up_ratio) |
| **mainline_state ENUM** | `CONFIRMED` (score≥65), `FORMING` (50-65), `EARLY` (30-50), `ROTATING` (15-30), `FADE` (<15) |
| **PK** | `(trade_date, mainline_id)` |
| **Indexes** | `idx_trade_date`, `idx_mainline_date` (mainline_id, trade_date), `idx_state_date` (mainline_state, trade_date) |
| **Run After** | `build_ga_stock_role_map_daily.py`, `build_cn_stock_mainline_strength_daily.py` |

### [`cn_ga_market_pulse_daily`](sql/ddl/ga_p0_mainline_context_schema.sql:52)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Daily market-wide pulse: aggregate market state, breadth metrics, and regime classification. Used by Factor 8 (risk_crowding_score) in the Unified Alpha Engine. |
| **Build Script** | [`scripts/build_ga_market_pulse_daily.py`](scripts/build_ga_market_pulse_daily.py) |
| **Sources** | `cn_ga_mainline_radar_daily`, `cn_index_daily_price` |
| **Key Alpha Columns** | `bullish_industry_ratio` (CONFIRMED+FORMING mainline fraction), `bearish_industry_ratio` (FADE fraction), `market_phase` (TREND_EXPANSION/DIFFUSION/BOTTOM_REPAIR/DIVERGENCE/TOP_DECAY/RISK_OFF/CRISIS) |
| **market_state ENUM** | `TREND_STRONG` (bullish>0.5), `TREND_WEAK` (bullish 0.3-0.5), `RANGE`, `RISK_OFF` (bearish>0.5) |
| **market_phase logic** | RISK_OFF/CRISIS trigger +0.2 risk penalty in Factor 8 |
| **PK** | `(trade_date)` |
| **Indexes** | `idx_trade_date`, `idx_state_date` (market_state, trade_date) |
| **Run After** | `build_ga_mainline_radar_daily.py` |

### [`cn_ga_stock_role_map_daily`](sql/ddl/ga_p0_mainline_context_schema.sql:60)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Daily mapping of each stock's role within its mainline. Provides `mainline_id` (critical bridge between stock and mainline for Factors 4/5/7/8) and `stock_role` (LEADER/CORE/MOMENTUM/NON_CORE). |
| **Build Script** | [`scripts/build_ga_stock_role_map_daily.py`](scripts/build_ga_stock_role_map_daily.py) |
| **Sources** | `cn_stock_leader_score_daily` only (full history 2010-01-04+) |
| **Key Alpha Columns** | `mainline_id` (= industry_id from leader_score), `stock_role` (LEADER/CORE/MOMENTUM/NON_CORE) |
| **stock_role ENUM** | `LEADER` (CORE_LEADER), `CORE` (NEAR_LEADER), `MOMENTUM` (EDGE_LEADER), `NON_CORE` (NON_LEADER) |
| **role_score** | (leader/3)*40 + rs_pct*35 + turn_pct*25 (0-100) |
| **PK** | `(trade_date, symbol)` |
| **Indexes** | `idx_trade_date`, `idx_symbol_date` (symbol, trade_date), `idx_role_date` (stock_role, trade_date), `idx_mainline_date` (mainline_id, trade_date) |
| **No Dependencies** | Self-contained from leader_score; run this FIRST among GA tables |

### [`cn_ga_market_context_daily`](sql/ddl/ga_p0_mainline_context_schema.sql:22)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Daily high-level market context summary: market regime, mainline phase, rotation state, trend confidence score, risk context, and narrative summary. |
| **PK** | `(trade_date)` |

### [`cn_ga_data_readiness_daily`](docs/DDL/cn_market_red_p0_mainline_data_fix.sql:156)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Data readiness audit log: tracks per-table row counts, max trade date, null rates, and status (READY/DEGRADED/STALE/MISSING) for each trade date. |
| **PK** | `(id)` auto-increment |
| **Unique** | `(trade_date, table_name)` |

---

## 6. State / Job Tracking

### [`cn_mainline_backfill_job_state`](scripts/build_local_industry_proxy_daily.py:403)
| Aspect | Description |
|--------|-------------|
| **Purpose** | Tracks backfill job progress across chunks. Enables `--resume` functionality: if a chunk is marked `completed`, it is skipped on re-run. |
| **Key Columns** | `job_name`, `chunk_key`, `range_start`, `range_end`, `status` (running/completed/failed), `attempts`, `last_rows`, `last_error`, `last_run_id` |
| **PK** | `(job_name, chunk_key)` |

---

## 7. Data Flow Diagram

```
cn_stock_daily_price  ──┐
cn_stock_daily_basic  ──┤
cn_stock_leader_score  ──┤
cn_local_industry_map_hist ──┘
        │
        ▼
cn_local_industry_proxy_daily  ◄── build_local_industry_proxy_daily.py
        │
        ▼
cn_industry_capital_flow_daily  ◄── industry_capital_flow.py
        │
        ▼
cn_mainline_strength_daily  ◄── build_cn_stock_mainline_strength_daily.py
        │                                                    ▲
        ▼                                                    │
cn_stock_leader_score_daily ────────────────────────────────┤
        │                                                    │
        └──► cn_ga_stock_role_map_daily  ◄── build_ga_stock_role_map_daily.py
                        │
                        ▼
        cn_ga_mainline_radar_daily  ◄── build_ga_mainline_radar_daily.py
                        │    (also reads: cn_stock_daily_price,
                        │                cn_stock_mainline_strength_daily)
                        ▼
        cn_ga_market_pulse_daily  ◄── build_ga_market_pulse_daily.py
                        │    (also reads: cn_index_daily_price)
                        ▼
        cn_ga_market_context_daily  (no build script)

cn_stock_income ──┐
cn_stock_balancesheet ──┤
cn_stock_fina_indicator ──┤
        │
        ▼
  cn_local_stock_income_q  ──┐
  cn_local_stock_balancesheet_q ──┤
  cn_local_stock_fina_indicator_q ──┤
        │
        ▼
  cn_stock_fundamental_daily  ◄── stock_fundamental_daily.py
        │
        └──► cn_mainline_strength_daily (earnings_score)
```

---

## 8. Collation Summary

Different tables use different collations. All `symbol`-based JOINs between these tables must use explicit `CONVERT(... USING utf8mb4) COLLATE utf8mb4_unicode_ci` to avoid `Illegal mix of collations` errors.

| Table | Collation |
|-------|-----------|
| `cn_stock_daily_price` | `utf8mb4_unicode_ci` |
| `cn_stock_daily_basic` | `utf8mb4_general_ci` |
| `cn_stock_leader_score_daily` | `utf8mb4_unicode_ci` |
| `cn_local_industry_map_hist` | `utf8mb4_0900_ai_ci` |
| `cn_local_industry_proxy_daily` | `utf8mb4_0900_ai_ci` |
| `cn_sw_industry_daily` | `utf8mb4_general_ci` |
| `cn_stock_income` | `utf8mb4_general_ci` |
| `cn_stock_balancesheet` | `utf8mb4_general_ci` |
| `cn_stock_fina_indicator` | `utf8mb4_general_ci` |
| `cn_local_stock_income_q` | `utf8mb4_unicode_ci` |
| `cn_local_stock_balancesheet_q` | `utf8mb4_unicode_ci` |
| `cn_local_stock_fina_indicator_q` | `utf8mb4_unicode_ci` |
| `cn_stock_fundamental_daily` | `utf8mb4_unicode_ci` |
