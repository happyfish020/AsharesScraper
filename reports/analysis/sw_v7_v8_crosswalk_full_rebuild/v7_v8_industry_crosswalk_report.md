# V7/V8 Industry Crosswalk Report

- start: `2000-01-04`
- end: `2026-05-15`
- srcs: `SW2021, SW2014`
- top_k: `3`

## Summary

- best rows: `103688`
- one_to_one: `67441`
- one_to_many: `0`
- unmatched: `8962`
- manual_review: `27285`

## By Source

- `SW2014` / `MANUAL_REVIEW`: `13395`
- `SW2014` / `ONE_TO_ONE`: `32608`
- `SW2014` / `UNMATCHED`: `5841`
- `SW2021` / `MANUAL_REVIEW`: `13890`
- `SW2021` / `ONE_TO_ONE`: `34833`
- `SW2021` / `UNMATCHED`: `3121`

## Review / Unmatched Examples

- `2010-01-29` `850112.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850112.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850135.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850135.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850136.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850136.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850325.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850325.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850326.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850326.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850339.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850339.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850344.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850344.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850521.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850521.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850772.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850772.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850781.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)
- `2010-01-29` `850781.SI` -> `None` (UNMATCHED, jaccard=0.0000, cov_v8=0.0000, reason=no_shared_symbols)

## Outputs

- candidates: `reports/analysis/sw_v7_v8_crosswalk_full_rebuild/v7_v8_industry_crosswalk.csv`
- best: `reports/analysis/sw_v7_v8_crosswalk_full_rebuild/v7_v8_industry_crosswalk_best.csv`
- one_to_many: `reports/analysis/sw_v7_v8_crosswalk_full_rebuild/v7_v8_industry_crosswalk_one_to_many.csv`
- unmatched: `reports/analysis/sw_v7_v8_crosswalk_full_rebuild/v7_v8_industry_crosswalk_unmatched.csv`
- manual_review: `reports/analysis/sw_v7_v8_crosswalk_full_rebuild/v7_v8_industry_crosswalk_manual_review.csv`
