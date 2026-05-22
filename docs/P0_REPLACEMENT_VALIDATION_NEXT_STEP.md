# P0 Replacement Validation Next Step

After full replacement-source pull finishes:

1. validate the loaded window
2. rebuild crosswalk from local DB tables
3. build latest best snapshot for replacement tests

## Commands

### 1. Validate actual loaded window

```powershell
python scripts\validate_replacement_window.py --expected-start 2023-01-01 --expected-end 2026-03-30
```

### 2. Rebuild crosswalk from DB sources

```powershell
python scripts\build_v7_v8_industry_crosswalk.py ^
  --start 2023-01-01 ^
  --end 2026-03-30 ^
  --replace ^
  --srcs SW2021 SW2014 ^
  --source-mode db ^
  --output-dir reports\analysis\sw_v7_v8_crosswalk_2023_20260330
```

### 3. Build latest best snapshot

```powershell
python scripts\build_v7_v8_crosswalk_latest.py --replace --output-dir reports\analysis\v7_v8_crosswalk_latest
```

This latest snapshot is the handoff artifact for trading-chain replacement validation.
