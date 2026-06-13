# P0H Mainline Lifecycle Rewire

This patch rewires `cn_mainline_lifecycle_daily` to consume only the metadata-driven fact layer:

- Source: `cn_mainline_strength_fact_daily`
- Target: `cn_mainline_lifecycle_daily`
- Forbidden source: `cn_ga_mainline_radar_daily`

## Build

```bat
python scripts/build_mainline_lifecycle_daily.py ^
  --start 2024-01-01 ^
  --end 2026-06-12 ^
  --dry-run ^
  --strict ^
  --db-password <password>
```

Then write:

```bat
python scripts/build_mainline_lifecycle_daily.py ^
  --start 2024-01-01 ^
  --end 2026-06-12 ^
  --replace ^
  --strict ^
  --db-password <password>
```

## Audit

```bat
python scripts/audit_mainline_lifecycle_fact_rewire.py ^
  --start 2024-01-01 ^
  --end 2026-06-12 ^
  --strict ^
  --db-password <password>
```

Expected:

- Daily `mainline_id` count = 41.
- `non_fact_rows = 0`.
- Fact and lifecycle coverage match.

## Notes

This patch intentionally replaces the legacy `scripts/build_mainline_lifecycle_daily.py` implementation. The legacy implementation treated `cn_ga_mainline_radar_daily` as the primary input. That is no longer allowed.
