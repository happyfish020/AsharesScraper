# V8 Tables Daily And Backfill Runbook

This runbook defines how to maintain the V8 replacement-source tables in `AsharesScraperV2`.

## Scope

Primary V8 replacement-source tables covered here:

- `cn_ts_sw_industry_master`
- `cn_ts_sw_industry_member_hist`
- `cn_v7_v8_industry_crosswalk`
- `cn_v7_v8_industry_crosswalk_latest`
- `cn_stock_v8_to_v7_sw_map_latest`

These tables are the data foundation for replacing V7 sector mapping with V8 local-industry mapping in a controlled way.

## Principles

- Do not mix V7 legacy board tables into this chain.
- Prefer local DB tables over live Tushare reads once source tables are materialized.
- Fail fast if required source tables are missing or empty.
- Always validate the loaded date window after a backfill run.

## Update Modes

### 1. Daily / Incremental Update

Use this mode after the newest market date has been loaded into the database.

Recommended sequence:

1. refresh official Tushare SW source tables
2. rebuild crosswalk from local DB source tables
3. rebuild latest snapshot tables

Commands:

```powershell
python scripts\build_tushare_sw_replacement_sources.py ^
  --start 2026-01-01 ^
  --end 2026-03-30 ^
  --srcs SW2021 SW2014 ^
  --levels L1 ^
  --replace-master ^
  --replace-members ^
  --output-dir reports\analysis\tushare_sw_replacement_sources_daily
```

```powershell
python scripts\build_v7_v8_industry_crosswalk.py ^
  --start 2026-01-01 ^
  --end 2026-03-30 ^
  --replace ^
  --srcs SW2021 SW2014 ^
  --source-mode db ^
  --output-dir reports\analysis\sw_v7_v8_crosswalk_daily
```

```powershell
python scripts\build_v7_v8_crosswalk_latest.py ^
  --replace ^
  --output-dir reports\analysis\v7_v8_crosswalk_latest
```

```powershell
python scripts\build_stock_v8_to_v7_sw_map_latest.py ^
  --replace ^
  --output-dir reports\analysis\stock_v8_to_v7_sw_map_latest
```

### 2. Historical Backfill

Use this mode when rebuilding the full V8 replacement-source chain.

Recommended sequence:

1. backfill official Tushare SW source tables
2. validate actual loaded window
3. rebuild crosswalk from local DB source tables
4. rebuild latest snapshot tables

Commands:

```powershell
python scripts\build_tushare_sw_replacement_sources.py ^
  --start 2023-01-01 ^
  --end 2026-03-30 ^
  --srcs SW2021 SW2014 ^
  --levels L1 ^
  --replace-master ^
  --replace-members ^
  --output-dir reports\analysis\tushare_sw_replacement_sources_2023_20260330
```

```powershell
python scripts\validate_replacement_window.py ^
  --expected-start 2023-01-01 ^
  --expected-end 2026-03-30
```

```powershell
python scripts\build_v7_v8_industry_crosswalk.py ^
  --start 2023-01-01 ^
  --end 2026-03-30 ^
  --replace ^
  --srcs SW2021 SW2014 ^
  --source-mode db ^
  --output-dir reports\analysis\sw_v7_v8_crosswalk_2023_20260330
```

```powershell
python scripts\build_v7_v8_crosswalk_latest.py ^
  --replace ^
  --output-dir reports\analysis\v7_v8_crosswalk_latest
```

```powershell
python scripts\build_stock_v8_to_v7_sw_map_latest.py ^
  --replace ^
  --output-dir reports\analysis\stock_v8_to_v7_sw_map_latest
```

## Expected Outputs

### Source Tables

- `cn_ts_sw_industry_master`
  - `SW2021 / L1`: expected `31` rows
  - `SW2014 / L1`: expected `28` rows

- `cn_ts_sw_industry_member_hist`
  - row count depends on Tushare membership history
  - current validated smoke:
    - `SW2021 / L1`: `5830`
    - `SW2014 / L1`: `5565`

### Crosswalk Tables

- `cn_v7_v8_industry_crosswalk`
  - monthly best-match plus candidates

- `cn_v7_v8_industry_crosswalk_latest`
  - latest best snapshot only

- `cn_stock_v8_to_v7_sw_map_latest`
  - stock-level latest mapped output for replacement validation

## Validation Checklist

After each run, check:

1. source summary markdown exists
2. crosswalk report markdown exists
3. latest snapshot CSV exists
4. stock-level latest map CSV exists
5. `validate_replacement_window.py` shows the intended range

## Current Known Reference Counts

Latest validated latest snapshot:

- `cn_v7_v8_industry_crosswalk_latest`
  - `2026-03-30`
  - `SW2014 = 258`
  - `SW2021 = 258`

- `cn_stock_v8_to_v7_sw_map_latest`
  - `2026-03-30`
  - `SW2014 = 4918`
  - `SW2021 = 4918`

## Related Docs

- [P0_TUSHARE_SW_REPLACEMENT_SOURCES.md](./P0_TUSHARE_SW_REPLACEMENT_SOURCES.md)
- [P0_SW_CROSSWALK_BUILDER.md](./P0_SW_CROSSWALK_BUILDER.md)
- [P0_REPLACEMENT_VALIDATION_NEXT_STEP.md](./P0_REPLACEMENT_VALIDATION_NEXT_STEP.md)
- [P0_STOCK_V8_TO_V7_LATEST_MAP.md](./P0_STOCK_V8_TO_V7_LATEST_MAP.md)
