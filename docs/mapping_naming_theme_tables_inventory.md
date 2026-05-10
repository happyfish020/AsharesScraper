# Mapping / Naming / Theme Tables Inventory

Checked against local `cn_market` on `2026-05-05`.

This note focuses on the data objects needed for:

- `concept_id -> concept_name`
- `trade_date + symbol -> concept_id`
- `trade_date + symbol -> industry_id / industry_name`
- industry / concept strength, state, rank, score
- stable inputs for `supercycle_tag` / `subtheme_tag`

## Quick Take

Best currently available objects by use case:

- Stock-to-concept mapping:
  - `cn_board_member_map_d`
- Concept name master:
  - `cn_board_concept_master`
  - current DB now contains both legacy `BKxxxx` rows and `TSxxx` rows loaded from Tushare concept master sync
- Stock-to-industry mapping:
  - `cn_board_member_map_d`
- Industry name master:
  - `cn_board_industry_master`
- Industry daily strength:
  - `cn_sw_industry_daily`
- Cross-sectional sector score / state / rank:
  - `cn_sector_rotation_ranked_t`
  - `cn_sector_rotation_signal_t`

Main gaps confirmed in current DB:

- `cn_board_member_map_d` concept rows on `2026-04-30` use `TSxxx`.
- `Tushare concept_detail` concept membership history is only supported from `2021-02-05` onward in the current upstream payload.
- pre-`2021-02-05` concept history should be treated as unsupported by this source, not as a load failure.
- `cn_stock_leader_score_v2` is a `VIEW`, not a physical table, and returns `0` rows for `2026-04-29` and `2026-04-30` in the current DB.

## Inventory

### 1. `cn_board_member_map_d`

- Object type: table
- Purpose:
  - daily historical stock-to-sector mapping
  - covers both concept and industry via `sector_type`
- Key fields:
  - `trade_date`
  - `sector_type`
  - `sector_id`
  - `symbol`
- Observed coverage:
  - all rows: `2000-01-04` to `2026-04-30`
  - `2026-04-30` concept rows: `34444`
  - `2026-04-30` industry rows: `7099`
- Sample meaning:
  - concept sample on `2026-04-30`:
    - `000001 -> TS354 / TS108 / TS145 / TS147 / TS143`
  - industry sample on `2026-04-30`:
    - `000001 -> 857831.SI`
    - `000002 -> 851811.SI`
- Fit for current need:
  - this is the best current source for `trade_date + symbol + concept_id`
  - this is also the best current source for `trade_date + symbol + industry_id`
- Gap:
  - `sector_id` is only an id; downstream still needs a reliable naming/master table

### 2. `cn_board_concept_master`

- Object type: table
- Purpose:
  - concept master / naming dictionary
- Key fields:
  - `CONCEPT_ID`
  - `CONCEPT_NAME`
  - `PROVIDER`
  - `ASOF_DATE`
  - `SOURCE`
- Observed coverage:
  - legacy `BKxxxx` naming rows from `AKSHARE`
  - `TSxxx` naming rows loaded from Tushare concept master sync
  - `TSxxx` coverage in local DB: `879` ids
- Sample meaning:
  - `BK0490 -> 军工`
  - `BK0493 -> 新能源`
  - source is `AKSHARE`
- Fit for current need:
  - usable for both `BKxxxx -> concept_name` and `TSxxx -> concept_name`
- Gap:
  - concept member history from the same Tushare family is not full-history; practical overlap begins at `2021-02-05`

### 3. `cn_board_industry_master`

- Object type: table
- Purpose:
  - industry code/name master
- Key fields:
  - `BOARD_ID`
  - `BOARD_NAME`
  - `PROVIDER`
  - `ASOF_DATE`
  - `SOURCE`
- Observed coverage:
  - rows: `117`
  - distinct ids: `117`
  - `2026-01-09` to `2026-04-17`
- Sample meaning:
  - `801010.SI -> 农林牧渔`
  - `801080.SI -> 电子`
  - provider/source sample: `TUSHARE / TUSHARE_SW2021_L1`
- Fit for current need:
  - usable as an industry name master
  - especially strong for SW2021 L1 naming
- Gap:
  - latest `cn_board_member_map_d` industry ids include values such as `857831.SI`
  - latest `cn_sector_rotation_ranked_t` still shows unresolved names such as `850815.SI`
  - so current master coverage is not yet sufficient for every live `.SI` id used downstream

### 4. `cn_sw_industry_daily`

- Object type: table
- Purpose:
  - Shenwan industry daily market/valuation/strength source
- Key fields:
  - `ts_code`
  - `trade_date`
  - `name`
  - `pct_change`
  - `pe`
  - `pb`
  - `float_mv`
- Observed coverage:
  - `2009-10-29` to `2026-04-30`
  - latest day row count: `31`
- Sample meaning:
  - `801010.SI 农林牧渔 pct_change=0.88`
  - `801080.SI 电子 pct_change=2.21`
- Fit for current need:
  - good source for `trade_date + industry_id + score-like fields`
  - especially suitable for L1 `industry_score` inputs
- Gap:
  - this is an industry daily fact table, not a stock mapping table
  - it does not solve concept naming or stock-to-concept mapping

### 5. `cn_sector_rotation_ranked_t`

- Object type: table
- Purpose:
  - daily sector state / score / rank result
- Key fields:
  - `TRADE_DATE`
  - `SECTOR_TYPE`
  - `SECTOR_ID`
  - `SECTOR_NAME`
  - `STATE`
  - `TIER`
  - `THEME_GROUP`
  - `THEME_RANK`
  - `SCORE`
- Observed coverage:
  - `2000-01-04` to `2026-04-30`
  - latest day row count: `1146`
- Sample meaning:
  - `2026-04-30 CONCEPT TS45 state=CONFIRM score=6.2140992200`
  - `2026-04-30 INDUSTRY 850815.SI state=CONFIRM score=4.6257481700`
- Fit for current need:
  - very useful for `trade_date + concept_id/industry_id + score/state/rank`
  - best current source for cross-sectional sector strength
- Gap:
  - naming is not stable when master mapping is missing
  - unresolved ids flow through as `sector_name = sector_id`

### 6. `cn_sector_rotation_signal_t`

- Object type: table
- Purpose:
  - daily rotation action / transition result
- Key fields:
  - `SIGNAL_DATE`
  - `SECTOR_TYPE`
  - `SECTOR_ID`
  - `SECTOR_NAME`
  - `ACTION`
  - `SCORE`
  - `STATE`
  - `TRANSITION`
- Observed coverage:
  - `2000-01-04` to `2026-04-30`
  - latest day row count: `1146`
- Sample meaning:
  - `2026-04-30 CONCEPT TS45 action=WATCH state=CONFIRM transition=DIRECT_CONFIRM`
  - `2026-04-30 INDUSTRY 850815.SI action=WATCH state=CONFIRM transition=NO_CHANGE`
- Fit for current need:
  - useful explainability layer for `sector_state`, `sector_rank` adjacent diagnostics
- Gap:
  - same naming problem as `cn_sector_rotation_ranked_t`

### 7. `cn_stock_leader_score_v2`

- Object type: view
- Purpose:
  - stock leader helper layer
- Current DB fact:
  - object type is `VIEW`
  - row count on `2026-04-29`: `0`
  - row count on `2026-04-30`: `0`
- Why this matters:
  - it should not be treated as the authoritative current mapping layer
  - it is currently not usable for late-April 2026 leader or naming joins in this DB
- Likely compatibility issue:
  - design doc says the view currently ranks only `sector_id LIKE 'BK%'`
  - current live board mapping on late-April 2026 is dominated by `TSxxx` and `.SI`
  - this mismatch likely explains the empty rows

## What To Use Right Now

If the strategy layer needs the fastest workable handoff today:

- Stock-to-concept:
  - use `cn_board_member_map_d`
- Concept naming:
  - currently missing for live `TSxxx`
  - do not rely on `cn_board_concept_master` for `TSxxx`
- Stock-to-industry:
  - use `cn_board_member_map_d`
- Industry naming:
  - use `cn_board_industry_master` where available
  - expect unresolved `.SI` ids for some live downstream rows
- Industry/concept strength:
  - use `cn_sector_rotation_ranked_t`
  - use `cn_sector_rotation_signal_t` for explainability
- L1 industry strength:
  - use `cn_sw_industry_daily`

## Recommended Interpretation

Current DB already has enough to support:

- historical `stock -> concept_id`
- historical `stock -> industry_id`
- daily sector score/state/rank
- SW L1 industry strength
- live `TSxxx -> concept_name`
- near-term concept membership / concept map from `2021-02-05+`

Current DB does not yet fully support:

- full-history concept membership before `2021-02-05` from Tushare
- stable L2 subtheme naming
- clean named outputs from `cn_sector_rotation_named_v`
- late-April 2026 `cn_stock_leader_score_v2` consumption
