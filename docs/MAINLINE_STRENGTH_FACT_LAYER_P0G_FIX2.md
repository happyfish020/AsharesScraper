# P0G Fix2 — Mainline Universe Normalization

P0G found `cn_mainline_strength_fact_daily` V1 was not a stable fact universe:

- 2026 latest daily mainline count: 31
- historical max daily mainline count: 331
- historical top rows mixed `850xxx.SI`, `857xxx.SI`, `BKxxxx`, and `801xxx.SI`

Therefore V1 is **not accepted** as GrowthAlpha Ground Truth. The issue is not row coverage; it is universe drift.

## Fix2

Adds a stable SW-L1 fact builder:

```bat
python scripts/build_cn_mainline_strength_fact_l1_daily.py --start 2024-01-01 --end 2026-06-12 --dry-run --strict
python scripts/build_cn_mainline_strength_fact_l1_daily.py --start 2024-01-01 --end 2026-06-12 --replace --strict
python scripts/audit_cn_mainline_strength_fact_daily.py --start 2024-01-01 --end 2026-06-12 --strict --max-daily-mainline-drift 3
```

This builder uses only:

- `cn_local_industry_proxy_daily`
- stable L1 universe: `industry_level='L1'` or `industry_id REGEXP '^801[0-9]{3}\\.SI$'`

It does **not** read `cn_ga_mainline_radar_daily`.

## Important

The previous `build_mainline_lifecycle_daily_from_fact.py --replace` run was based on failed P0G V1. Do not let GrowthAlpha consume it.

Optional quarantine:

```bat
python scripts/quarantine_fact_lifecycle_write.py --start 2024-01-01 --end 2026-06-12
python scripts/quarantine_fact_lifecycle_write.py --start 2024-01-01 --end 2026-06-12 --apply
```

After L1 fact audit passes, rerun lifecycle from the normalized fact table.
