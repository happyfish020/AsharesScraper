# Data Asset Audit Report -- `cn_market_red`
**Generated**: 2026-05-08 19:54:39
**As-Of Date**: 2026-05-08
**Tables Audited**: 20

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **Total Tables** | 20 |
| **Healthy (OK)** | 12 |
| **Warning (WARN/STALE/MISSING_COLUMNS)** | 8 |
| **Failed (FAIL/MISSING_TABLE/EMPTY)** | 0 |
| **P0 Critical Failures** | 0 |

> :warning: **Some tables have warnings. Review recommended.**

## 2. Critical Table Status

| Table | Tier | Status | Severity | Rows | Max Date | Lag (d) |
|-------|------|--------|----------|------|----------|---------|
| `cn_stock_daily_price` | P0_CRITICAL | OK | CRITICAL | 16,717,406 | 2026-05-06 | 2 |
| `cn_index_daily_price` | P0_CRITICAL | OK | CRITICAL | 29,182 | 2026-05-06 | 2 |
| `cn_sw_industry_daily` | P0_CRITICAL | OK | CRITICAL | 102,090 | 2026-05-06 | 2 |
| `cn_stock_monthly_basic` | P0_CRITICAL | OK | CRITICAL | 737,805 | 2026-04-30 | 8 |
| `cn_stock_fina_indicator` | P0_CRITICAL | WARN | CRITICAL | 255,045 | 2026-03-31 | 38 |
| `cn_stock_daily_basic` | P1_IMPORTANT | OK | WARN | 16,808,419 | 2026-05-06 | 2 |
| `cn_stock_leader_score_daily` | P1_IMPORTANT | STALE | WARN | 11,938,438 | 2026-04-30 | 8 |
| `cn_stock_leader_sw_l1_latest_snap` | P1_IMPORTANT | OK | WARN | 15,501 | 2026-05-06 | 2 |
| `cn_stock_universe_status_t` | P1_IMPORTANT | OK | WARN | 6,186 | 2026-05-06 | 2 |

## 3. Per-Tier Detail

### P0 -- Critical (Core Market Data)

| Table | Status | Rows | Min Date | Max Date | Lag (d) | Missing Columns | Null Rates |
|-------|--------|------|----------|----------|---------|-----------------|------------|
| `cn_stock_daily_price` | OK | 16,717,406 | 2000-01-04 | 2026-05-06 | 2 | -- | SYMBOL=0.0%; TRADE_DATE=0.0%; OPEN=0.0%; HIGH=0.0%; LOW=0.0%; CLOSE=0.0%; VOLUME=0.0%; AMOUNT=0.0%; CHG_PCT=0.00% |
| `cn_index_daily_price` | OK | 29,182 | 2010-01-04 | 2026-05-06 | 2 | -- | INDEX_CODE=0.0%; TRADE_DATE=0.0%; OPEN=0.00%; CLOSE=0.0%; HIGH=0.00%; LOW=0.00%; VOLUME=0.0%; AMOUNT=0.0%; PRE_CLOSE=0.03%; CHG_PCT=0.00% |
| `cn_sw_industry_daily` | OK | 102,090 | 2009-10-29 | 2026-05-06 | 2 | -- | ts_code=0.0%; name=0.0%; trade_date=0.0%; close=0.0%; amount=0.0% |
| `cn_stock_monthly_basic` | OK | 737,805 | 2008-01-31 | 2026-04-30 | 8 | -- | symbol=0.0%; trade_date=0.0%; month_key=0.0%; total_mv=0.0%; circ_mv=0.0%; pe_ttm=17.07%; pb=0.77% |
| `cn_stock_fina_indicator` | WARN | 255,045 | 2008-03-31 | 2026-03-31 | 38 | -- | symbol=0.0%; end_date=0.0%; ann_date=0.00%; report_type=100.00%; netprofit_yoy=4.97%; q_profit_yoy=11.91%; or_yoy=5.27%; eps=4.33%; roe=1.74%; grossprofit_margin=3.04% |

### P1 -- Important (Derived / Quality)

| Table | Status | Rows | Min Date | Max Date | Lag (d) | Missing Columns | Null Rates |
|-------|--------|------|----------|----------|---------|-----------------|------------|
| `cn_stock_daily_basic` | OK | 16,808,419 | 2000-01-04 | 2026-05-06 | 2 | -- | symbol=0.0%; trade_date=0.0%; pe_ttm=16.55%; pb=0.88%; total_mv=0.0%; circ_mv=0.0%; volume_ratio=0.15% |
| `cn_stock_leader_score_daily` | STALE | 11,938,438 | 2010-01-04 | 2026-04-30 | 8 | -- | trade_date=0.0%; symbol=0.0%; leader_score=0.0%; leader_bucket=0.0% |
| `cn_stock_leader_sw_l1_latest_snap` | OK | 15,501 | 2026-02-27 | 2026-05-06 | 2 | -- | trade_date=0.0%; symbol=0.0%; leader_score=0.0%; sw_l1_id=0.0% |
| `cn_stock_universe_status_t` | OK | 6,186 | 2001-04-13 | 2026-05-06 | 2 | -- | symbol=0.0%; is_active=0.0%; last_trade_date=0.0% |

### P2 -- Structure (Mapping / Proxy)

| Table | Status | Rows | Min Date | Max Date | Lag (d) | Missing Columns | Null Rates |
|-------|--------|------|----------|----------|---------|-----------------|------------|
| `cn_board_member_map_d` | OK | 41,098,956 | 2000-01-04 | 2026-05-06 | 2 | -- | trade_date=0.0%; sector_type=0.0%; sector_id=0.0%; symbol=0.0% |
| `cn_local_industry_map_hist` | MISSING_COLUMNS | 20,349 | 1991-01-29 | 2026-04-17 | 21 | in_date, out_date, is_current | symbol=0.0%; industry_id=0.0% |
| `cn_local_industry_proxy_daily` | STALE | 990,208 | 2010-01-04 | 2026-04-29 | 9 | -- | industry_id=0.0%; trade_date=0.0%; member_count=0.0%; ret_eqw=0.0%; amount_total=1.71% |

### P3 -- Reporting (Financial Events)

| Table | Status | Rows | Min Date | Max Date | Lag (d) | Missing Columns | Null Rates |
|-------|--------|------|----------|----------|---------|-----------------|------------|
| `cn_stock_income` | OK | 287,061 | 1996-12-31 | 2026-03-31 | 38 | -- | symbol=0.0%; end_date=0.0%; ann_date=0.0%; report_type=0.0%; total_revenue=0.24%; n_income_attr_p=0.09% |
| `cn_stock_balancesheet` | OK | 263,774 | 2001-12-31 | 2026-03-31 | 38 | -- | symbol=0.0%; end_date=0.0%; ann_date=0.0%; report_type=0.0%; total_assets=0.01%; total_liab=0.15% |
| `cn_event_disclosure_date` | OK | 266,112 | 2008-03-31 | 2026-03-31 | 38 | -- | symbol=0.0%; end_date=0.0%; pre_date=1.13%; actual_date=3.33% |
| `cn_event_earnings_forecast` | OK | 114,941 | 2008-01-02 | 2026-04-29 | 9 | -- | symbol=0.0%; ann_date=0.0%; end_date=0.0%; forecast_type=0.0%; p_change_min=7.43%; p_change_max=8.26% |

### GA Layer (GrowthAlpha Engine)

| Table | Status | Rows | Min Date | Max Date | Lag (d) | Missing Columns | Null Rates |
|-------|--------|------|----------|----------|---------|-----------------|------------|
| `cn_ga_mainline_radar_daily` | STALE | 26,739 | 2026-01-05 | 2026-04-30 | 8 | -- | trade_date=0.0%; mainline_id=0.0%; mainline_name=0.0%; member_count=0.0%; leader_count=0.0%; mainline_score=0.0%; mainline_state=0.0%; rank_no=0.0%; reason=0.0% |
| `cn_ga_market_pulse_daily` | STALE | 77 | 2026-01-05 | 2026-04-30 | 8 | -- | trade_date=0.0%; market_score=0.0%; market_state=0.0%; target_exposure=0.0%; breadth_up_ratio=0.0%; risk_flag=68.83%; reason=0.0% |
| `cn_ga_stock_role_map_daily` | STALE | 5,175 | 2026-02-27 | 2026-02-27 | 70 | -- | trade_date=0.0%; symbol=0.0%; stock_name=32.17%; mainline_id=0.0%; mainline_name=0.0%; leader_score=0.0%; stock_role=0.0%; role_score=0.0%; role_reason=0.0% |
| `cn_ga_data_readiness_daily` | WARN | 25 |  |  | N/A | -- | trade_date=100.00%; table_name=0.0%; status=0.0%; severity=100.00%; row_count=0.0%; max_trade_date=4.00%; null_rate_summary=100.00% |

## 4. Date Coverage Summary

| Table | Tier | Min Date | Max Date | Distinct Days | Lag (d) | Threshold (d) |
|-------|------|----------|----------|---------------|---------|---------------|
| `cn_stock_daily_price` | P0_CRITICAL | 2000-01-04 | 2026-05-06 | 6,379 | 2 | 5 |
| `cn_index_daily_price` | P0_CRITICAL | 2010-01-04 | 2026-05-06 | 3,964 | 2 | 5 |
| `cn_sw_industry_daily` | P0_CRITICAL | 2009-10-29 | 2026-05-06 | 4,010 | 2 | 5 |
| `cn_stock_monthly_basic` | P0_CRITICAL | 2008-01-31 | 2026-04-30 | 224 | 8 | 45 |
| `cn_stock_fina_indicator` | P0_CRITICAL | 2008-03-31 | 2026-03-31 | 95 | 38 | 90 |
| `cn_stock_daily_basic` | P1_IMPORTANT | 2000-01-04 | 2026-05-06 | 6,379 | 2 | 5 |
| `cn_stock_leader_score_daily` | P1_IMPORTANT | 2010-01-04 | 2026-04-30 | 3,960 | 8 | 5 |
| `cn_stock_leader_sw_l1_latest_snap` | P1_IMPORTANT | 2026-02-27 | 2026-05-06 | 3 | 2 | 5 |
| `cn_stock_universe_status_t` | P1_IMPORTANT | 2001-04-13 | 2026-05-06 | 258 | 2 | N/A |
| `cn_board_member_map_d` | P2_STRUCTURE | 2000-01-04 | 2026-05-06 | 6,379 | 2 | 7 |
| `cn_local_industry_map_hist` | P2_STRUCTURE | 1991-01-29 | 2026-04-17 | 2,362 | 21 | N/A |
| `cn_local_industry_proxy_daily` | P2_STRUCTURE | 2010-01-04 | 2026-04-29 | 3,947 | 9 | 5 |
| `cn_stock_income` | P3_REPORTING | 1996-12-31 | 2026-03-31 | 97 | 38 | N/A |
| `cn_stock_balancesheet` | P3_REPORTING | 2001-12-31 | 2026-03-31 | 77 | 38 | N/A |
| `cn_event_disclosure_date` | P3_REPORTING | 2008-03-31 | 2026-03-31 | 73 | 38 | N/A |
| `cn_event_earnings_forecast` | P3_REPORTING | 2008-01-02 | 2026-04-29 | 4,296 | 9 | N/A |
| `cn_ga_mainline_radar_daily` | GA_LAYER | 2026-01-05 | 2026-04-30 | 74 | 8 | 5 |
| `cn_ga_market_pulse_daily` | GA_LAYER | 2026-01-05 | 2026-04-30 | 77 | 8 | 5 |
| `cn_ga_stock_role_map_daily` | GA_LAYER | 2026-02-27 | 2026-02-27 | 1 | 70 | 5 |
| `cn_ga_data_readiness_daily` | GA_LAYER |  |  | 0 | N/A | N/A |

## 5. Missing Columns

| Table | Tier | Missing Columns |
|-------|------|-----------------|
| `cn_local_industry_map_hist` | P2_STRUCTURE | `in_date, out_date, is_current` |

## 6. Stale Tables

| Table | Tier | Max Date | Lag (d) | Threshold (d) |
|-------|------|----------|---------|---------------|
| `cn_stock_leader_score_daily` | P1_IMPORTANT | 2026-04-30 | 8 | 5 |
| `cn_local_industry_proxy_daily` | P2_STRUCTURE | 2026-04-29 | 9 | 5 |
| `cn_ga_mainline_radar_daily` | GA_LAYER | 2026-04-30 | 8 | 5 |
| `cn_ga_market_pulse_daily` | GA_LAYER | 2026-04-30 | 8 | 5 |
| `cn_ga_stock_role_map_daily` | GA_LAYER | 2026-02-27 | 70 | 5 |

## 7. Null Rate Analysis

Tables with high null rates on key columns (>20%):

- `cn_stock_fina_indicator` (P0_CRITICAL): symbol=0.0%; end_date=0.0%; ann_date=0.00%; report_type=100.00%; netprofit_yoy=4.97%; q_profit_yoy=11.91%; or_yoy=5.27%; eps=4.33%; roe=1.74%; grossprofit_margin=3.04%
- `cn_ga_data_readiness_daily` (GA_LAYER): trade_date=100.00%; table_name=0.0%; status=0.0%; severity=100.00%; row_count=0.0%; max_trade_date=4.00%; null_rate_summary=100.00%

## 8. Recommended Next Actions

| Priority | Table | Issue | Recommendation |
|----------|-------|-------|----------------|
| 1. | `cn_stock_fina_indicator` | WARN | High null rates on key columns: report_type=100.0%. |
| 2. | `cn_stock_leader_score_daily` | STALE | Latest data is 8 days behind as-of date (threshold: 5d). |
| 3. | `cn_local_industry_map_hist` | MISSING_COLUMNS | Missing key columns: in_date, out_date, is_current. |
| 4. | `cn_local_industry_proxy_daily` | STALE | Latest data is 9 days behind as-of date (threshold: 5d). |
| 5. | `cn_ga_mainline_radar_daily` | STALE | Latest data is 8 days behind as-of date (threshold: 5d). |
| 6. | `cn_ga_market_pulse_daily` | STALE | Latest data is 8 days behind as-of date (threshold: 5d). |
| 7. | `cn_ga_stock_role_map_daily` | STALE | Latest data is 70 days behind as-of date (threshold: 5d). |
| 8. | `cn_ga_data_readiness_daily` | WARN | High null rates on key columns: trade_date=100.0%, severity=100.0%, null_rate_summary=100.0%. |

## 9. Readiness Assessment

### Mainline Strength Engine
**Status**: NOT READY
**Blockers**: `cn_local_industry_map_hist` (MISSING_COLUMNS); `cn_local_industry_proxy_daily` (STALE); `cn_ga_mainline_radar_daily` (STALE)

### Market Breadth Engine
**Status**: NOT READY
**Blockers**: `cn_ga_market_pulse_daily` (STALE)

### Narrative / Context Layer
**Status**: NOT READY
**Blockers**: `cn_ga_mainline_radar_daily` (STALE); `cn_ga_market_pulse_daily` (STALE); `cn_ga_stock_role_map_daily` (STALE); `cn_stock_fina_indicator` (WARN)

---
*Report generated by `audit_cn_market_data_assets.py` at 2026-05-08 19:54:39*