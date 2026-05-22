# V7/V8 Industry Crosswalk Report

- start: `2026-01-01`
- end: `2026-03-30`
- srcs: `SW2021, SW2014`
- top_k: `3`

## Summary

- best rows: `1548`
- one_to_one: `1500`
- one_to_many: `0`
- unmatched: `48`
- manual_review: `0`

## By Source

- `SW2014` / `ONE_TO_ONE`: `726`
- `SW2014` / `UNMATCHED`: `48`
- `SW2021` / `ONE_TO_ONE`: `774`

## Review / Unmatched Examples

- `2026-01-30` `859511.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-01-30` `859512.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-01-30` `859521.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-01-30` `859621.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-01-30` `859622.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-01-30` `859631.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-01-30` `859632.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-01-30` `859633.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-01-30` `859711.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-01-30` `859712.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-01-30` `859713.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-01-30` `859714.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-01-30` `859721.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-01-30` `859811.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-01-30` `859821.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-01-30` `859822.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-02-27` `859511.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-02-27` `859512.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-02-27` `859521.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2026-02-27` `859621.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)

## Outputs

- candidates: `reports/analysis/sw_v7_v8_crosswalk_smoke_2026q1_db/v7_v8_industry_crosswalk.csv`
- best: `reports/analysis/sw_v7_v8_crosswalk_smoke_2026q1_db/v7_v8_industry_crosswalk_best.csv`
- one_to_many: `reports/analysis/sw_v7_v8_crosswalk_smoke_2026q1_db/v7_v8_industry_crosswalk_one_to_many.csv`
- unmatched: `reports/analysis/sw_v7_v8_crosswalk_smoke_2026q1_db/v7_v8_industry_crosswalk_unmatched.csv`
- manual_review: `reports/analysis/sw_v7_v8_crosswalk_smoke_2026q1_db/v7_v8_industry_crosswalk_manual_review.csv`
