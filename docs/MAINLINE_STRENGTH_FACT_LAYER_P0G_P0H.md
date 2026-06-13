# Mainline Strength Fact Layer P0G/P0H

## Purpose

This delivery keeps `cn_ga_mainline_radar_daily` alive but prevents the next layer from depending on it blindly.

The new source of truth remains:

```text
cn_mainline_strength_fact_daily
```

It is built without reading `cn_ga_mainline_radar_daily`.

## Added scripts

```text
scripts/audit_cn_mainline_strength_fact_daily.py
scripts/build_mainline_lifecycle_daily_from_fact.py
```

## P0G audit

Run after the fact table build/validation:

```bat
python scripts/audit_cn_mainline_strength_fact_daily.py --start 2024-01-01 --end 2026-06-12 --strict
```

It checks:

```text
coverage
NULL ratios for rs_60d / rs_120d
daily mainline-count drift
quality flags
checkpoint Top-N mainlines
optional radar comparison, read-only
```

## P0H lifecycle rewire, safe mode

Dry run first:

```bat
python scripts/build_mainline_lifecycle_daily_from_fact.py --start 2024-01-01 --end 2026-06-12 --dry-run
```

Then write:

```bat
python scripts/build_mainline_lifecycle_daily_from_fact.py --start 2024-01-01 --end 2026-06-12 --replace
python scripts/validate_mainline_lifecycle_daily.py --start 2024-01-01 --end 2026-06-12 --min-rows 1
```

## V8 task switch

Default behavior remains legacy lifecycle builder.

To use fact-based lifecycle in V8 chain:

```bat
set V8_USE_FACT_LIFECYCLE=1
```

To skip the extra fact audit:

```bat
set V8_RUN_MAINLINE_STRENGTH_FACT_AUDIT=0
```

## Important rule

`build_mainline_lifecycle_daily_from_fact.py` does not read `cn_ga_mainline_radar_daily`.

Radar replacement should proceed in this order:

```text
1. Fact table audit PASS
2. Lifecycle from Fact PASS
3. Market Pulse rewire
4. Unified Alpha rewire
5. GrowthAlpha consumer rewire
6. Radar downgraded to display cache
```
