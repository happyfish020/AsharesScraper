# CN Event Tables Quick Guide

This note summarizes the `cn_event_*` tables used by the event-driven data layer in `cn_market`.

## Overview

- All raw event tables carry `source`, `raw_payload`, and `updated_at` for auditability.
- Primary keys are designed for idempotent upserts.
- `cn_event_signal_daily` is a standardized, derived table for strategy research.

## Tables

1. `cn_event_earnings_forecast`
   - Purpose: earnings pre-announcement (业绩预告)
   - Primary key: `(symbol, ann_date, end_date, forecast_type, report_type)`
   - Key fields: `forecast_type`, `p_change_min/max`, `net_profit_min/max`, `summary`
   - DDL: [cn_event_earnings_forecast.sql](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/docs/DDL/cn_market.cn_event_earnings_forecast.sql)

1. `cn_event_earnings_express`
   - Purpose: earnings express report (业绩快报)
   - Primary key: `(symbol, ann_date, end_date)`
   - Key fields: `revenue`, `n_income`, `diluted_eps`, `yoy_*`
   - DDL: [cn_event_earnings_express.sql](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/docs/DDL/cn_market.cn_event_earnings_express.sql)

1. `cn_event_fina_indicator`
   - Purpose: financial indicator announcements (财务指标披露)
   - Primary key: `(symbol, ann_date, end_date, report_type)`
   - Key fields: `eps`, `roe`, `roa`, `grossprofit_margin`, `debt_to_eqt`, `netprofit_yoy`, `q_sales_yoy`, `q_profit_yoy`
   - DDL: [cn_event_fina_indicator.sql](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/docs/DDL/cn_market.cn_event_fina_indicator.sql)

1. `cn_event_disclosure_date`
   - Purpose: planned/actual disclosure dates (披露日期计划/实际)
   - Primary key: `(symbol, end_date)`
   - Key fields: `pre_date`, `actual_date`, `modify_date`
   - DDL: [cn_event_disclosure_date.sql](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/docs/DDL/cn_market.cn_event_disclosure_date.sql)

1. `cn_event_dividend`
   - Purpose: dividend plan/approval/execution (分红送股)
   - Primary key: `(symbol, ann_date, end_date, div_proc)`
   - Key fields: `record_date`, `ex_date`, `pay_date`, `stk_div`, `cash_div`, `div_proc`
   - DDL: [cn_event_dividend.sql](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/docs/DDL/cn_market.cn_event_dividend.sql)

1. `cn_event_announcement_meta`
   - Purpose: announcement metadata (公告元数据, optional by Tushare权限)
   - Primary key: `(symbol, ann_date, title)`
   - Key fields: `title`, `url`, `type`
   - DDL: [cn_event_announcement_meta.sql](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/docs/DDL/cn_market.cn_event_announcement_meta.sql)

1. `cn_event_signal_daily`
   - Purpose: standardized event signals for research/backtest
   - Primary key: `(trade_date, symbol, event_type, event_subtype, raw_event_id, version)`
   - Key fields: `event_type`, `event_subtype`, `event_score`, `event_direction`, `anchor_date`
   - DDL: [cn_event_signal_daily.sql](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/docs/DDL/cn_market.cn_event_signal_daily.sql)

## Notes

- `raw_payload` stores the original Tushare row JSON for reproducibility.
- If `cn_event_announcement_meta` is empty, the usual cause is Tushare权限不足 for `anns_d`.

## Validation Checks

For `forecast` and `express`, `ann_date` should be the announcement date and typically later than `end_date` (report period end). A simple check:

```sql
SELECT
  'cn_event_earnings_forecast' AS table_name,
  COUNT(*) AS total_rows,
  SUM(ann_date < end_date) AS ann_before_end
FROM cn_event_earnings_forecast
UNION ALL
SELECT
  'cn_event_earnings_express' AS table_name,
  COUNT(*) AS total_rows,
  SUM(ann_date < end_date) AS ann_before_end
FROM cn_event_earnings_express;
```
