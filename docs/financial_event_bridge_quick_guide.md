# Financial Event Bridge Quick Guide

This view is intended for downstream strategy projects that want one stable read path.

## View

- `cn_stock_financial_event_bridge_v1`

## Priority

- official statement first:
  - `cn_stock_income`
  - `cn_stock_fina_indicator`
- event-side supplements:
  - `cn_event_disclosure_date`
  - `cn_event_earnings_express`
  - `cn_event_earnings_forecast`

## Core Fields

- `symbol`
- `end_date`
- `ann_date`
- `f_ann_date`
- `visible_ann_date`
- `n_income_attr_p`
- `q_profit_yoy`
- `netprofit_yoy`
- `q_sales_yoy`
- `or_yoy`

## Suggested Filter Example

```sql
SELECT
    symbol,
    end_date,
    visible_ann_date,
    n_income_attr_p,
    q_profit_yoy,
    netprofit_yoy
FROM cn_stock_financial_event_bridge_v1
WHERE end_date = '2026-03-31'
  AND n_income_attr_p > 3000000
  AND visible_ann_date IS NOT NULL;
```

## Notes

- one row per `symbol`, using the latest available official report period in `cn_stock_income`
- if disclosure planning exists, `visible_ann_date` prefers official announcement dates and then disclosure dates
- `express_*` and forecast fields are supplements, not replacements for the official statement
