# GCRL CN Public Fund V1 Delivery

## Changed Files

- `docs/DDL/cn_market.cn_gcrl_public_fund.sql`
- `docs/gcrl/GCRL_CN_PUBLIC_FUND_V1_DESIGN.md`
- `app/tools/sync_cn_gcrl_public_fund_from_tushare.py`
- `app/tasks/gcrl_cn_public_fund_task.py`
- `scripts/gcrl/run_gcrl_cn_public_fund_collector.py`
- `scripts/gcrl/audit_gcrl_cn_public_fund_schema.py`
- `app/cli.py`

## New MySQL Tables

All tables use the required `cn_` prefix:

- `cn_gcrl_institution_registry`
- `cn_gcrl_fund_registry`
- `cn_gcrl_position_snapshot`
- `cn_gcrl_position_change`
- `cn_gcrl_data_freshness`

## Run

```bash
python scripts/gcrl/run_gcrl_cn_public_fund_collector.py --report-period 20250331
```

or:

```bash
set GCRL_CN_REPORT_PERIOD=20250331
python -m app.cli --tasks gcrl_cn_public_fund --asof 20250331 --no-vpn
```

## Boundary

This delivery is AshareScraper only. It collects and stores domestic public fund facts. It does not do theme mapping, reference score, consensus score, or buy/sell signals.
