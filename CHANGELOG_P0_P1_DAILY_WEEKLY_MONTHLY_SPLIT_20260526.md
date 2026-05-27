# P0/P1 Daily-Weekly-Monthly Maintenance Split Changelog

## Modified files

- `daily_spot_update.bat`
- `weekly.bat`
- `monthly.bat`
- `monthly_full_map_backfill.bat` new helper
- `app/tasks/v8_dataset_ops_task.py`
- `docs/V8_DAILY_WEEKLY_MONTHLY_MAINTENANCE_MATRIX_20260526.md`

## Key changes

1. Daily default changed from full financial-quality-alpha refresh to light operational refresh.
2. Daily skips `stock_fundamental_daily`, `stock_quality_score`, and `unified_alpha` by default.
3. Weekly now refreshes financial-quality and unified alpha after market audit.
4. Monthly now supports incremental map-history refresh by default and full map backfill only when explicitly requested.
5. Monthly failure path now returns real `%ERRORLEVEL%` instead of false success.
6. `_run_derived_alpha_chain()` now honors `V8_SKIP_UNIFIED_ALPHA=1`.

## Validation performed

- `python3 -m py_compile app/tasks/v8_dataset_ops_task.py`

Windows `.bat` files were statically reviewed only in this environment; run them once in Windows CMD/PowerShell for live verification.
