# GCRL CN Public Fund V1.1 Update

Scope: update files only.

Changes:

1. Tushare `fund_portfolio` is no longer the primary holding source.
2. EastMoney/Tiantian Fund quarterly holding archive is now the default holding source.
3. Tushare `fund_basic` remains the registry source.
4. `fund_portfolio` permission failure is optional and does not break the run.
5. Added global NaN/NaT/inf -> None cleanup before MySQL writes.
6. Added `cn_gcrl_data_source_status` source health table.
7. Output tables remain unchanged:
   - `cn_gcrl_position_snapshot`
   - `cn_gcrl_position_change`

Run:

```bash
python scripts/gcrl/run_gcrl_cn_public_fund_collector.py --report-period 20250331
```

Smoke test with limited funds:

```bash
python scripts/gcrl/run_gcrl_cn_public_fund_collector.py --report-period 20250331 --holding-source eastmoney --eastmoney-max-funds 50
```

Registry only:

```bash
python scripts/gcrl/run_gcrl_cn_public_fund_collector.py --report-period 20250331 --holding-source none
```

Optional Tushare holding source:

```bash
python scripts/gcrl/run_gcrl_cn_public_fund_collector.py --report-period 20250331 --holding-source tushare
```
