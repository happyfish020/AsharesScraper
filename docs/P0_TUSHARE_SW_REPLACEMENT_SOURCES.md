# P0 Tushare SW Replacement Sources

This task builds the official Tushare Shenwan source tables that V8 replacement work can depend on without reusing legacy V7 board tables.

## Tables

- `cn_ts_sw_industry_master`
- `cn_ts_sw_industry_member_hist`

## Source APIs

- `index_classify`
- `index_member_all`

## Why New Tables

These tables are intentionally separate from:

- `cn_board_industry_master`
- `cn_board_industry_member_hist`
- `cn_board_member_map_d`

This avoids mixing legacy V7 board data with new V8 replacement-source data.

## Run

### Smoke Test

```powershell
python scripts\build_tushare_sw_replacement_sources.py ^
  --start 2026-01-01 ^
  --end 2026-03-30 ^
  --srcs SW2021 SW2014 ^
  --levels L1 ^
  --replace-master ^
  --replace-members ^
  --output-dir reports\analysis\tushare_sw_replacement_sources_smoke_2026q1
```

### Full Pull

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

## Outputs

- `tushare_sw_replacement_sources_summary.csv`
- `tushare_sw_replacement_sources_summary.md`

## Expected Use Order

1. Build official Tushare SW source tables
2. Build / refresh V7-V8 crosswalk with `--source-mode db`
3. Validate zero-drift replacement path before switching any trading-chain reader
