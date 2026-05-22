# P0 SW Crosswalk Builder

This task builds the first read-only crosswalk needed before replacing V7 sector rotation inputs with V8 local-industry inputs.

## Goal

Generate a reproducible mapping between:

- V7 SW L1 code: `801xxx.SI`
- V8 local industry code: mostly `85xxxx.SI`
- industry names
- parent fields when available
- `src = SW2021 / SW2014`
- monthly `effective_date`

The builder uses:

1. existing `cn_local_industry_map_hist`
2. local `cn_ts_sw_industry_master`
3. local `cn_ts_sw_industry_member_hist`

Recommended upstream source task:

- [`docs/P0_TUSHARE_SW_REPLACEMENT_SOURCES.md`](./P0_TUSHARE_SW_REPLACEMENT_SOURCES.md)

## Outputs

DB table:

- `cn_v7_v8_industry_crosswalk`

Files:

- `v7_v8_industry_crosswalk.csv`
- `v7_v8_industry_crosswalk_best.csv`
- `v7_v8_industry_crosswalk_one_to_many.csv`
- `v7_v8_industry_crosswalk_unmatched.csv`
- `v7_v8_industry_crosswalk_manual_review.csv`
- `v7_v8_industry_crosswalk_report.md`

## Mapping Rules

Per monthly `effective_date`, the builder compares:

- one V8 local industry stock set
- against each Tushare SW L1 stock set

Metrics:

- `shared_symbol_count`
- `jaccard_score`
- `coverage_vs_v8`
- `coverage_vs_v7`

Classification:

- `ONE_TO_ONE`
- `ONE_TO_MANY`
- `UNMATCHED`
- `MANUAL_REVIEW`

## Run

```powershell
python scripts\build_v7_v8_industry_crosswalk.py ^
  --start 2023-01-01 ^
  --end 2026-03-30 ^
  --replace ^
  --srcs SW2021 SW2014 ^
  --source-mode db
```

Direct Tushare mode is still available for debugging:

```powershell
python scripts\build_v7_v8_industry_crosswalk.py --start 2023-01-01 --end 2026-03-30 --replace --srcs SW2021 SW2014 --source-mode tushare
```

## Why This Comes First

Without this crosswalk, replacing V7 sector rotation inputs with V8 local industry tables will keep drifting because:

- V7 often uses official SW L1 `801xxx.SI`
- V8 local maps currently contain many `85xxxx.SI`
- the same concept may exist under different code systems

This builder is the required bridge before:

1. changing trading-chain sector mapping
2. backfilling new Tushare tables for production replacement
