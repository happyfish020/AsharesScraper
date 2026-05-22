# V7/V8 Industry Crosswalk Report

- start: `2023-01-01`
- end: `2026-03-30`
- srcs: `SW2021, SW2014`
- top_k: `3`

## Summary

- best rows: `20124`
- one_to_one: `19500`
- one_to_many: `0`
- unmatched: `624`
- manual_review: `0`

## By Source

- `SW2014` / `ONE_TO_ONE`: `9438`
- `SW2014` / `UNMATCHED`: `624`
- `SW2021` / `ONE_TO_ONE`: `10062`

## Review / Unmatched Examples

- `2023-01-31` `859511.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-01-31` `859512.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-01-31` `859521.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-01-31` `859621.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-01-31` `859622.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-01-31` `859631.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-01-31` `859632.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-01-31` `859633.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-01-31` `859711.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-01-31` `859712.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-01-31` `859713.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-01-31` `859714.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-01-31` `859721.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-01-31` `859811.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-01-31` `859821.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-01-31` `859822.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-02-28` `859511.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-02-28` `859512.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-02-28` `859521.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2023-02-28` `859621.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)

## Outputs

- candidates: `reports/analysis/sw_v7_v8_crosswalk_2023_20260330/v7_v8_industry_crosswalk.csv`
- best: `reports/analysis/sw_v7_v8_crosswalk_2023_20260330/v7_v8_industry_crosswalk_best.csv`
- one_to_many: `reports/analysis/sw_v7_v8_crosswalk_2023_20260330/v7_v8_industry_crosswalk_one_to_many.csv`
- unmatched: `reports/analysis/sw_v7_v8_crosswalk_2023_20260330/v7_v8_industry_crosswalk_unmatched.csv`
- manual_review: `reports/analysis/sw_v7_v8_crosswalk_2023_20260330/v7_v8_industry_crosswalk_manual_review.csv`
