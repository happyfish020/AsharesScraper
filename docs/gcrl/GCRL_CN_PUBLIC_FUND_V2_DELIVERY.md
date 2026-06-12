# GCRL CN Public Fund V2 Delivery

## Scope

AshareScraper data-production layer only.

- EastMoney / Tiantian Fund holding archive is the primary holding source.
- Tushare `fund_portfolio` is removed from the execution path.
- Tushare `fund_basic` is optional only for fund registry enrichment. If no token is available, the collector uses existing `cn_gcrl_fund_registry` or EastMoney registry fallback.
- No theme mapping, no score, no buy/sell signal.

## Default Run

```bash
python scripts/gcrl/run_gcrl_cn_public_fund_collector.py
```

Default period is the latest completed public fund reporting period.

## Backfill

```bash
python scripts/gcrl/run_gcrl_cn_public_fund_collector.py --report-period 20250331
python scripts/gcrl/run_gcrl_cn_public_fund_collector.py --year 2024
python scripts/gcrl/run_gcrl_cn_public_fund_collector.py --start-period 20230331 --end-period 20250331
python scripts/gcrl/run_gcrl_cn_public_fund_collector.py --backfill-all
```

## Debug

```bash
python scripts/gcrl/run_gcrl_cn_public_fund_collector.py --report-period 20250331 --max-funds 50
python scripts/gcrl/run_gcrl_cn_public_fund_collector.py --report-period 20250331 --allow-empty
```

`--allow-empty` is only for source debugging. Normal runs fail when `snapshot_rows == 0`.

## Tables

- `cn_gcrl_institution_registry`
- `cn_gcrl_fund_registry`
- `cn_gcrl_position_snapshot`
- `cn_gcrl_position_change`
- `cn_gcrl_data_freshness`
- `cn_gcrl_data_source_status`
