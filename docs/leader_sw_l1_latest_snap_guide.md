# Leader SW L1 Latest Snap Guide

This document covers the materialized latest snapshot table for:

- `cn_stock_leader_score_v2`
- SW L1 stock mapping
- `cn_sw_industry_daily`

The table is:

- `cn_stock_leader_sw_l1_latest_snap`

## Why This Exists

`cn_stock_leader_score_v2` is a full-history view and is slow for ad-hoc joins.

The previously added view:

- `cn_stock_leader_sw_l1_latest_v`

is logically correct, but still inherits the performance profile of `cn_stock_leader_score_v2`.

So this table materializes one latest safe trading date into a normal InnoDB table.

## Safe Latest Date Rule

If `--trade-date` is not provided, the builder chooses:

```sql
LEAST(
  MAX(cn_stock_daily_basic.trade_date),
  MAX(cn_board_member_map_d.trade_date WHERE sector_type='INDUSTRY'),
  MAX(cn_sw_industry_daily.trade_date)
)
```

This keeps the snapshot aligned with the slowest dependency.

## Objects

DDL:

- [cn_market.cn_stock_leader_sw_l1_latest_snap.sql](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/docs/DDL/cn_market.cn_stock_leader_sw_l1_latest_snap.sql)

Builder script:

- [build_cn_stock_leader_sw_l1_latest_snap.py](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/app/tools/build_cn_stock_leader_sw_l1_latest_snap.py)

## Usage

Build latest safe date automatically:

```powershell
python -m app.tools.build_cn_stock_leader_sw_l1_latest_snap
```

Build a fixed trade date:

```powershell
python -m app.tools.build_cn_stock_leader_sw_l1_latest_snap --trade-date 2026-02-27
```

## Stored Fields

The snapshot stores:

- stock leader fields from `cn_stock_leader_score_v2`
- current BK industry id/name from leader source
- SW L1 id/name from `cn_board_industry_member_hist`
- same-date SW daily metrics from `cn_sw_industry_daily`

Important columns:

- `trade_date`
- `symbol`
- `stock_name`
- `leader_score`
- `leader_bucket`
- `rs_percentile`
- `turnover_20d_percentile`
- `bk_industry_id`
- `bk_industry_name`
- `sw_l1_id`
- `sw_l1_name`
- `sw_close`
- `sw_pct_change`
- `sw_pe`
- `sw_pb`

## Example Queries

Strong leaders with SW L1 context:

```sql
SELECT symbol, stock_name, leader_score, leader_bucket, sw_l1_name, sw_pct_change
FROM cn_stock_leader_sw_l1_latest_snap
WHERE trade_date = (SELECT MAX(trade_date) FROM cn_stock_leader_sw_l1_latest_snap)
  AND leader_score >= 2
ORDER BY rs_percentile DESC, turnover_20d_percentile DESC
LIMIT 50;
```

Top leaders by industry:

```sql
SELECT sw_l1_name, COUNT(*) AS cnt
FROM cn_stock_leader_sw_l1_latest_snap
WHERE trade_date = (SELECT MAX(trade_date) FROM cn_stock_leader_sw_l1_latest_snap)
  AND leader_score >= 2
GROUP BY sw_l1_name
ORDER BY cnt DESC, sw_l1_name;
```

Industry-aware stock review:

```sql
SELECT symbol, stock_name, leader_score, leader_bucket, sw_l1_name, sw_pct_change, sw_pe, sw_pb
FROM cn_stock_leader_sw_l1_latest_snap
WHERE trade_date = (SELECT MAX(trade_date) FROM cn_stock_leader_sw_l1_latest_snap)
ORDER BY leader_score DESC, sw_pct_change DESC, symbol
LIMIT 100;
```
