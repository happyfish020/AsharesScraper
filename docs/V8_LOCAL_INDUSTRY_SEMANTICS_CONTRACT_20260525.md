# V8 Local Industry Semantics Contract

Date: 2026-05-25

## Purpose

This document defines the industry-level semantics used by the V8 production
path in `AsharesScraperV2`.

The goal is to align V8 around what can be built reliably from Tushare Pro
under the current operating constraint of a 2000-point account:

- Tushare SW classification metadata is available
- Tushare SW constituent membership is available
- official `sw_daily` is not a guaranteed primary source at 2000 points
- therefore V8 production should treat locally-computed proxy industry series
  as the primary daily industry source

This contract is V8-first. It does not preserve V7 naming conventions as a
source of truth.

## Production Rule

V8 should reason about two industry sets:

1. `SW_L1`
   - official Shenwan 2021 level-1 industries
   - coarse industry dictionary
   - current observed count: 31 industries
   - best used for coarse market structure, reference, and compatibility views

2. `LOCAL_FINE`
   - V8 fine-grained local industry set built from stock constituents and daily
     stock data
   - current observed count: 391 industries
   - this is the primary V8 production industry layer for proxy, breadth,
     strength, radar, lifecycle, and alpha inputs

## Current Physical Storage Mapping

The current database stores the two production sets with legacy level labels:

| Physical Table | Physical `industry_level` | Semantic Set | Observed Industry Count | Notes |
| --- | --- | --- | ---: | --- |
| `cn_local_industry_map_hist` | `L3` | `LOCAL_FINE` | 391 | This is the stock-to-industry membership history used to build the fine proxy layer. |
| `cn_local_industry_map_hist` | `SW_L1` | `SW_L1` | 31 | This is the official Shenwan level-1 membership layer. |
| `cn_local_industry_proxy_daily` | `L1` | `LOCAL_FINE` | 391 | Legacy label only. Semantically this is the V8 fine-grained proxy layer, not official SW L1. |
| `cn_local_industry_proxy_daily` | `SW_L1` | `SW_L1` | 31 | Coarse official Shenwan level-1 proxy layer. |

## Interpretation Rules

The following rules apply until a later physical rename/migration is completed:

1. `cn_local_industry_proxy_daily.industry_level = 'L1'` must be interpreted as
   `LOCAL_FINE`, not as official SW level-1.
2. `cn_local_industry_map_hist.industry_level = 'L3'` is the membership source
   for `LOCAL_FINE`.
3. `SW_L1` keeps its literal meaning across both tables.
4. For V8 production logic, semantic meaning is more important than the legacy
   string stored in `industry_level`.

## Tushare-Driven Design Rationale

With a 2000-point Tushare Pro account, V8 can reliably access:

- `index_classify` for SW hierarchy metadata
- `index_member_all` for SW constituent membership
- `daily` for stock daily prices
- `daily_basic` for stock daily market-cap and turnover metrics

This is sufficient to build:

- `cn_local_industry_map_hist`
- `cn_local_industry_proxy_daily`

This is not sufficient to make official `sw_daily` the sole required industry
price source for the V8 primary path. Therefore V8 should continue to use the
local proxy path as the primary daily industry source.

## V8 Usage Guidance

### Primary daily production layer

Use `LOCAL_FINE` for:

- `cn_local_industry_proxy_daily`
- `cn_industry_capital_flow_daily`
- `cn_stock_mainline_strength_daily`
- `cn_ga_mainline_radar_daily`
- `cn_mainline_lifecycle_daily`
- `cn_unified_alpha_score_daily`

### Secondary coarse layer

Use `SW_L1` for:

- coarse market structure summaries
- compatibility views
- reference-level dashboards
- optional crosswalk and reporting context

## Migration Direction

This contract does not require an immediate table rewrite.

Near-term priority:

- standardize documentation and code comments around `LOCAL_FINE`
- stop reading proxy `L1` as if it were official SW L1
- stop assuming `cn_local_industry_map_hist` must contain physical `L1` rows for
  the fine-grained V8 production path

Longer-term optional cleanup:

- add a semantic layer or compatibility view that exposes `LOCAL_FINE`
- gradually replace legacy `L1` references in V8 code with semantic names
- preserve `SW_L1` as the only literal official level label in the production
  contract
