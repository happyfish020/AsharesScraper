# P0G1 Metadata-Driven Mainline Registry

## Decision

Use the existing DB metadata layer as the only GrowthAlpha standard ID system:

```text
cn_meta_mainline_registry
cn_meta_subline_registry
cn_meta_stock_mainline_map
cn_meta_subline_stock_map
```

Do not create a parallel `cn_canonical_*` registry.

## Added table

```text
cn_meta_source_mainline_map
```

Purpose: map external source codes into system-owned `mainline_id` values before any consumer reads them.

Examples:

```text
SW / L1 / 801080.SI -> CN_ELECTRONICS
SW / L1 / 801770.SI -> CN_COMMUNICATION
SW / L1 / 801780.SI -> CN_BANK
```

## New scripts

```text
scripts/seed_cn_meta_mainline_full_registry.py
scripts/audit_cn_meta_mainline_full_registry.py
scripts/build_cn_mainline_strength_fact_meta_daily.py
```

## Run order

Preview metadata seed:

```bat
python scripts/seed_cn_meta_mainline_full_registry.py --dry-run --strict
```

Write metadata seed:

```bat
python scripts/seed_cn_meta_mainline_full_registry.py --apply --strict
```

Audit metadata:

```bat
python scripts/audit_cn_meta_mainline_full_registry.py --strict
```

Build metadata-driven Fact Layer dry-run:

```bat
python scripts/build_cn_mainline_strength_fact_meta_daily.py --start 2024-01-01 --end 2026-06-12 --dry-run --strict
```

If dry-run passes:

```bat
python scripts/build_cn_mainline_strength_fact_meta_daily.py --start 2024-01-01 --end 2026-06-12 --replace --strict
```

## Rules

```text
1. GrowthAlpha must consume cn_meta mainline_id, never raw 801/850/BK codes.
2. External source codes are allowed only in cn_meta_source_mainline_map.
3. cn_ga_mainline_radar_daily remains legacy and must not feed this layer.
4. Strategic themes remain stock-map driven via cn_meta_stock_mainline_map.
5. SW-L1 market sectors remain source-map driven via cn_meta_source_mainline_map.
```

## Standard ID shape

Market sectors use `CN_*` IDs, for example:

```text
CN_BANK
CN_ELECTRONICS
CN_COMMUNICATION
CN_POWER_EQUIPMENT_SECTOR
CN_MACHINERY
```

Strategic GrowthAlpha themes keep the existing IDs, for example:

```text
OPTICAL_COMMS
PCB_ELECTRONICS
AI_COMPUTE
AI_POWER_INFRA
AI_DEEPER_INFRA
ROBOTICS_AUTOMATION
```
