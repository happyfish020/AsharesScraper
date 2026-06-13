# P0G-0 Canonical Mainline Registry

## Purpose

External source IDs are unstable and mixed:

- SW L1: `801xxx.SI`
- SW L2/L3: `850xxx.SI`, `857xxx.SI`
- EastMoney board: `BKxxxx`
- Custom themes: project-defined strings

GrowthAlpha should not consume those raw IDs directly. This delivery introduces a canonical ID layer.

```text
external source code
  -> cn_source_mainline_code_map
  -> cn_canonical_mainline_registry
  -> canonical MAINLINE_ID
  -> cn_mainline_strength_fact_daily
  -> GrowthAlpha
```

## New tables

### cn_canonical_mainline_registry

System-owned canonical mainline registry.

Primary key:

```text
canonical_mainline_id
```

Example:

```text
ML_CN_SWL1_801080
ML_CN_SWL1_801770
ML_CN_SWL1_801730
```

### cn_source_mainline_code_map

Maps external source codes into canonical IDs.

Primary key:

```text
source_system, source_code, effective_start_date
```

## New scripts

```text
scripts/build_cn_canonical_mainline_registry.py
scripts/audit_cn_canonical_mainline_registry.py
scripts/build_cn_mainline_strength_fact_canonical_daily.py
```

## Important rule

These scripts do **not** read `cn_ga_mainline_radar_daily`.

## Run order

```bat
python scripts/build_cn_canonical_mainline_registry.py --start 2024-01-01 --end 2026-06-12 --replace-sw-l1 --strict
python scripts/audit_cn_canonical_mainline_registry.py --start 2024-01-01 --end 2026-06-12 --strict
python scripts/build_cn_mainline_strength_fact_canonical_daily.py --start 2024-01-01 --end 2026-06-12 --dry-run --strict
```

If dry-run passes:

```bat
python scripts/build_cn_mainline_strength_fact_canonical_daily.py --start 2024-01-01 --end 2026-06-12 --replace --strict
python scripts/audit_cn_mainline_strength_fact_daily.py --start 2024-01-01 --end 2026-06-12 --strict --max-daily-mainline-drift 3
```

## Expected result

`cn_mainline_strength_fact_daily.mainline_id` should become canonical system IDs, not raw source IDs.

Expected latest examples:

```text
ML_CN_SWL1_801080  电子
ML_CN_SWL1_801770  通信
ML_CN_SWL1_801730  电力设备
```

## Current scope

This version seeds canonical mappings from stable SW-L1 proxy data only. Later versions can add SW-L2/SW-L3/BK/custom-theme mappings, but every source must map through `cn_source_mainline_code_map` first.
