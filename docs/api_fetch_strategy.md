# API Fetch Strategy

## Goal

Keep each loader aligned with the official API access pattern instead of forcing one loop shape across all endpoints.

## Rules

### 1. Symbol-first

Use `loop symbol` when the upstream API is designed around `ts_code` history per security.

Current examples in this project:

- `income`
- `balancesheet`
- `fina_indicator`
- `forecast`
- `express`
- `dividend`

Reason:

- these endpoints naturally return one security's historical statement / event history
- Tushare docs for `forecast` and `express` explicitly describe the standard endpoint as single-stock history first

## 2. Date-first

Use `loop trade_date` or `loop ann_date` when the upstream API is designed as a market-wide snapshot.

Current examples in this project:

- `daily_basic`
- some event-side market-wide announcement style fetches

Reason:

- `daily_basic` official docs explicitly recommend date-based extraction for full history
- these interfaces are naturally aligned to one market date returning many securities
- **Hybrid (Daily Price):** For `stock_daily_price`, the system uses a smart switch:
  - If `days > 1`: Uses **Symbol-first** to fill historical gaps per security.
  - If `days == 1`: 
    1. Audits gaps.
    2. If only the latest day is missing market-wide, uses **Date-first** (calling Tushare `daily` without `ts_code`) for 100x speedup.
    3. If specific symbols lag behind, uses **Symbol-first** for those only.

## 3. Period-first

## 3. Period-first

Use `loop period` when the upstream API is fundamentally quarterly / monthly by reporting period.

Current examples in this project:

- `inst_fund_hold_summary` via `fund_portfolio(period=...)`

## Current Project Mapping

- `stock_fundamental`
  - `monthly_basic`: date-first
  - `income`: symbol-first
  - `balancesheet`: symbol-first
  - `fina_indicator`: symbol-first
- `event`
  - `forecast`: symbol-first
  - `express`: symbol-first
  - `dividend`: symbol-first
  - `disclosure_date`: date/month-first
  - `anns_d`: date/month-first
- `stock_basic`
  - `daily_basic`: date-first
- `inst_fund_hold_summary`
  - `fund_portfolio`: period-first

## Operational Guidance

- do not force all loaders into `symbol-first`
- choose the loop dimension that matches the official API contract and pagination behavior
- when in doubt:
  - security history -> `symbol-first`
  - market snapshot -> `date-first`
  - reporting batch -> `period-first`


用法：
--tasks event_daily：只跑 forecast / express
--tasks event_periodic：跑 fina_indicator / disclosure / dividend
--tasks event：保留兼容，全跑
