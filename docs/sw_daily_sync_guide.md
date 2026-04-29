# SW Daily Sync Guide

This document covers the local MySQL landing for Tushare Pro `sw_daily`.

Reference:

- [Tushare Pro sw_daily](https://tushare.pro/document/2?doc_id=327)

Checked on:

- `2026-04-18`

## Goal

Store Shenwan industry daily行情 in a dedicated MySQL table, independent from broad-index daily prices.

The landed table is:

- `cn_sw_industry_daily`

This table is intended for:

- Shenwan industry daily price history
- industry relative-strength analytics
- future joins with SW L1 classification and rotation signals

## Why A New Table

We do not reuse `cn_index_daily_price` for `sw_daily`.

Reason:

- `sw_daily` has Shenwan-specific valuation fields like `pe`, `pb`, `float_mv`
- the semantic domain is industry行情, not generic broad indexes
- a separate table avoids mixing field conventions and source assumptions

## Table

DDL file:

- [cn_market.cn_sw_industry_daily.sql](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/docs/DDL/cn_market.cn_sw_industry_daily.sql)

Primary key:

- `(ts_code, trade_date)`

Important fields:

- `ts_code`
- `trade_date`
- `name`
- `open`
- `high`
- `low`
- `close`
- `change`
- `pct_change`
- `vol`
- `amount`
- `pe`
- `pb`
- `float_mv`
- `source`

## Sync Script

Script:

- [sync_cn_sw_industry_daily_from_tushare.py](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/app/tools/sync_cn_sw_industry_daily_from_tushare.py)

Behavior:

1. resolves Tushare token using the project's existing token resolver
2. ensures `cn_sw_industry_daily` exists
3. gets SW L1 codes from `cn_board_industry_master`
4. falls back to Tushare `index_classify(level='L1', src='SW2021')` if needed
5. fetches daily行情 from `pro.sw_daily(...)`
6. upserts rows into MySQL

Rate-limit handling:

- `sw_daily` currently has a per-minute access limit
- the sync script now auto-sleeps and retries when Tushare returns the 10-calls-per-minute limit error

## Usage

Full history:

```powershell
python -m app.tools.sync_cn_sw_industry_daily_from_tushare `
  --start 2000-01-01 `
  --end 2026-04-17 `
  --src SW2021 `
  --master-source TUSHARE_SW2021_L1 `
  --full
```

Incremental refresh:

```powershell
python -m app.tools.sync_cn_sw_industry_daily_from_tushare `
  --start 2026-01-01 `
  --end 2026-04-17 `
  --src SW2021 `
  --master-source TUSHARE_SW2021_L1
```

Single code debug:

```powershell
python -m app.tools.sync_cn_sw_industry_daily_from_tushare `
  --codes 801010.SI `
  --start 2026-01-01 `
  --end 2026-04-17 `
  --full
```

Optional token/config override:

```powershell
python -m app.tools.sync_cn_sw_industry_daily_from_tushare `
  --start 2026-01-01 `
  --end 2026-04-17 `
  --token YOUR_TOKEN
```

## Validation SQL

Row and date coverage:

```sql
SELECT COUNT(*) AS row_cnt,
       COUNT(DISTINCT ts_code) AS code_cnt,
       MIN(trade_date) AS min_trade_date,
       MAX(trade_date) AS max_trade_date
FROM cn_sw_industry_daily;
```

Latest day snapshot:

```sql
SELECT ts_code, name, trade_date, close, pct_change, pe, pb, float_mv
FROM cn_sw_industry_daily
WHERE trade_date = (SELECT MAX(trade_date) FROM cn_sw_industry_daily)
ORDER BY ts_code;
```

Join with SW L1 master:

```sql
SELECT d.ts_code, d.name, d.trade_date, d.close, m.BOARD_NAME
FROM cn_sw_industry_daily d
LEFT JOIN cn_board_industry_master m
  ON m.BOARD_ID = d.ts_code
 AND m.SOURCE = 'TUSHARE_SW2021_L1'
WHERE d.trade_date = (SELECT MAX(trade_date) FROM cn_sw_industry_daily)
ORDER BY d.ts_code;
```

## Recommended Next Use

After `cn_sw_industry_daily` is filled, the next low-cost analytics step is:

1. compute SW L1 relative strength by date
2. compare industry close / pct_change against stock leader-score distributions
3. decide whether rotation should switch from `BK%` taxonomy to SW L1

## Views Added

DDL files:

- [cn_market.cn_sw_industry_daily_latest_v.sql](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/docs/DDL/cn_market.cn_sw_industry_daily_latest_v.sql)
- [cn_market.cn_sw_industry_strength_latest_v.sql](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/docs/DDL/cn_market.cn_sw_industry_strength_latest_v.sql)
- [cn_market.cn_stock_sw_l1_latest_v.sql](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/docs/DDL/cn_market.cn_stock_sw_l1_latest_v.sql)
- [cn_market.cn_stock_leader_sw_l1_latest_v.sql](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/docs/DDL/cn_market.cn_stock_leader_sw_l1_latest_v.sql)

Purpose:

- `cn_sw_industry_daily_latest_v`
  - latest SW industry行情快照
- `cn_sw_industry_strength_latest_v`
  - latest SW industry强弱排序
- `cn_stock_sw_l1_latest_v`
  - latest stock to SW L1 mapping without waiting for `cn_board_member_map_d`
- `cn_stock_leader_sw_l1_latest_v`
  - latest leader-score snapshot joined with SW L1 and same-date SW daily metrics

Example latest strength query:

```sql
SELECT ts_code, name, close, pct_change, rs_20d, rs_20d_percentile, rs_rank, trend_state
FROM cn_sw_industry_strength_latest_v
ORDER BY rs_rank
LIMIT 20;
```

Example latest leader join query:

```sql
SELECT symbol, stock_name, leader_score, leader_bucket, sw_l1_name, sw_pct_change, sw_pe
FROM cn_stock_leader_sw_l1_latest_v
WHERE leader_score >= 2
ORDER BY rs_percentile DESC, turnover_20d_percentile DESC
LIMIT 50;
```
