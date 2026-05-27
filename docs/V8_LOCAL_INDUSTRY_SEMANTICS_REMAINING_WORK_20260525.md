# V8 Local Industry Semantics Remaining Work

Date: 2026-05-25

This note records the known remaining code references after the initial V8
semantic alignment pass.

## Updated in this pass

The following V8 production-path scripts were updated to follow the current
semantic contract:

- `scripts/build_local_industry_proxy_daily.py`
- `scripts/build_cn_stock_mainline_strength_daily.py`
- `scripts/build_cn_mainline_strength_daily.py`
- `scripts/build_industry_capital_flow_daily.py`

## Remaining known references

### 1. SW daily backfill script

File:

- `scripts/backfill_sw_industry_daily.py`

Notes:

- still reads `industry_level = 'L1'`
- this is tied to official SW hierarchy and `sw_daily`
- not a blocker for the current V8 local-proxy primary path

### 2. V7/V8 crosswalk builder

File:

- `data_pipeline/builders/sw_v7_v8_crosswalk.py`

Notes:

- now anchored to `cn_local_industry_map_hist.industry_level = 'L3'`
- still uses a compatibility subset filter (`85%.SI`)
- should continue to be treated as a compatibility/reporting layer, not as the
  definition of the full V8 `LOCAL_FINE` production universe

## Suggested next order

1. review `sw_v7_v8_crosswalk.py`
2. decide whether crosswalk should keep targeting the current compatibility
   subset or expand to the full `LOCAL_FINE` universe
3. only then adjust compatibility and reporting paths
