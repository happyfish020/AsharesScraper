# GCRL CN Public Fund Collector V1

Project: AshareScraper  
Module: GCRL_CN_PUBLIC_FUND  
Scope: 国内公募基金机构持仓事实采集层

## Boundary

AshareScraper only does:

- collect data from Tushare Pro
- normalize fund/institution/position fields
- write MySQL fact tables
- derive quarter-over-quarter position changes as factual deltas
- record data freshness

AshareScraper must not do:

- MAINLINE_ID / MAINLINE_NAME theme mapping
- theme flow score
- consensus score
- reference score
- BUY / SELL signal
- DailyOperationalReport rendering

Those belong to GrowthAlpha V8 consumer layers.

## Table Prefix Rule

All new tables use `cn_` prefix:

- `cn_gcrl_institution_registry`
- `cn_gcrl_fund_registry`
- `cn_gcrl_position_snapshot`
- `cn_gcrl_position_change`
- `cn_gcrl_data_freshness`

DDL:

- `docs/DDL/cn_market.cn_gcrl_public_fund.sql`

## First Batch Institutions

V1 seeds the following PUBLIC_FUND institutions:

- 易方达
- 华夏基金
- 嘉实基金
- 富国基金
- 南方基金
- 汇添富
- 景顺长城
- 广发基金
- 招商基金
- 博时基金
- 中欧基金
- 兴证全球
- 银华基金
- 工银瑞信
- 交银施罗德

## Data Source

Tushare Pro:

- `fund_basic`
- `fund_portfolio`

## Run Examples

Direct collector:

```bash
python scripts/gcrl/run_gcrl_cn_public_fund_collector.py --report-period 20250331
```

Dry run:

```bash
python scripts/gcrl/run_gcrl_cn_public_fund_collector.py --report-period 20250331 --dry-run
```

Runner task:

```bash
set GCRL_CN_REPORT_PERIOD=20250331
python -m app.cli --tasks gcrl_cn_public_fund --asof 20250331 --no-vpn
```

Schema audit:

```bash
python scripts/gcrl/audit_gcrl_cn_public_fund_schema.py
```

## Output Meaning

`cn_gcrl_position_snapshot` stores raw quarter-end fund holdings mapped to domestic public fund institutions.

`cn_gcrl_position_change` stores factual quarter-over-quarter changes:

- `NEW_POSITION`
- `ADD_POSITION`
- `REDUCE_POSITION`
- `EXIT_POSITION`
- `UNCHANGED`

These are not trading signals.
