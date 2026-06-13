# P0I Market Pulse / Unified Alpha Fact Rewire

## Goal

Remove `cn_ga_mainline_radar_daily` from AShareScraper downstream analytics.

`cn_ga_mainline_radar_daily` remains a legacy/display cache only. It must not be used as a primary source for market pulse, unified alpha, lifecycle, ranking, or GrowthAlpha report inputs.

## Updated files

```text
scripts/build_ga_market_pulse_daily.py
scripts/build_unified_alpha_score_daily.py
scripts/audit_radar_consumers.py
scripts/audit_ga_market_pulse_fact_rewire.py
scripts/audit_unified_alpha_fact_rewire.py
```

## Source of truth

```text
cn_meta_mainline_registry
cn_meta_source_mainline_map
cn_meta_stock_mainline_map
cn_meta_subline_registry
cn_meta_subline_stock_map
        ↓
cn_mainline_strength_fact_daily
        ↓
cn_mainline_lifecycle_daily
cn_ga_market_pulse_daily
cn_unified_alpha_score_daily
```

## What changed

### Market Pulse

Old:

```text
cn_ga_mainline_radar_daily → cn_ga_market_pulse_daily
```

New:

```text
cn_mainline_strength_fact_daily → cn_ga_market_pulse_daily
```

The builder writes `source_layer='FACT_META'`.

### Unified Alpha

Old:

```text
cn_ga_mainline_radar_daily → leader_dominance_score / risk_crowding_score
```

New:

```text
cn_mainline_strength_fact_daily → leader_dominance_score / risk_crowding_score
```

The factor functions still use the same in-memory variable names in some comments, but the SQL table dependency is removed.

## Run order

```bat
python scripts/audit_radar_consumers.py --root .

python scripts/build_ga_market_pulse_daily.py ^
  --start 2024-01-01 ^
  --end 2026-06-12 ^
  --dry-run ^
  --strict ^
  --db-password sec_Bobo123

python scripts/build_ga_market_pulse_daily.py ^
  --start 2024-01-01 ^
  --end 2026-06-12 ^
  --replace ^
  --strict ^
  --db-password sec_Bobo123

python scripts/audit_ga_market_pulse_fact_rewire.py ^
  --start 2024-01-01 ^
  --end 2026-06-12 ^
  --strict ^
  --db-password sec_Bobo123
```

Then rebuild Unified Alpha:

```bat
python scripts/build_unified_alpha_score_daily.py ^
  --start 2024-01-01 ^
  --end 2026-06-12 ^
  --dry-run ^
  --db-user cn_opr_red ^
  --db-password sec_Bobo123
```

If dry-run succeeds:

```bat
python scripts/build_unified_alpha_score_daily.py ^
  --start 2024-01-01 ^
  --end 2026-06-12 ^
  --replace ^
  --db-user cn_opr_red ^
  --db-password sec_Bobo123

python scripts/audit_unified_alpha_fact_rewire.py ^
  --start 2024-01-01 ^
  --end 2026-06-12 ^
  --strict ^
  --db-password sec_Bobo123
```

## Acceptance

```sql
select source_layer, count(*)
from cn_ga_market_pulse_daily
group by source_layer;
```

Expected: `FACT_META` only for 2024+ rebuilt range.

```bat
python scripts/audit_radar_consumers.py --root . --strict
```

Expected: no remaining non-allowlisted references in AShareScraper code.
