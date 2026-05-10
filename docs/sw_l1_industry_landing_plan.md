# SW L1 Industry Landing Plan

This document proposes a low-cost landing plan for Shenwan Level-1 industry data in local MySQL.

Current date context of this assessment:

- checked on `2026-04-18`
- current price/index data latest date: `2026-04-17`
- current board/industry mapping latest date: `2026-02-27`

## Goal

Replace the current coarse or Eastmoney-only industry usage with a stable Shenwan Level-1 industry chain that can support:

- stock to industry mapping by `trade_date`
- industry-level aggregation
- leader-score grouping
- future industry-index ingestion

Priority target:

- Shenwan Level-1 (`SW2021`, `L1`)

Optional fallback if source access is limited:

- keep existing Eastmoney `BK%` chain as backup

## Current Findings

### Available now

- `cn_board_industry_master`
- `cn_board_industry_cons`
- `cn_board_industry_member_hist`
- `cn_board_member_map_d`
- `app/tools/backfill_sw_l3_history_from_tushare.py`

### What is in MySQL now

- `cn_board_industry_master` currently stores Eastmoney industry boards only
  - sample ids: `BK0420`, `BK0477`
  - current provider/source: `EASTMONEY` / `AKSHARE`
- `cn_board_industry_member_hist` already contains one Shenwan history import
  - source label: `tushare_sw_l3`
  - ids like `850111.SI`
  - this is SW Level-3, not Level-1
- `cn_index_daily_price` does not currently contain Shenwan industry indexes
  - only broad indexes like `sh000300`, `sz399006`, `sh000688`

### Constraint

The existing downstream chain already depends on:

- `cn_board_industry_master`
- `cn_board_industry_member_hist`
- `cn_board_member_map_d`

So the lowest-cost path is:

1. keep using the existing table family
2. add SW L1 data into the same family
3. separate taxonomies by `PROVIDER` / `SOURCE` / id pattern
4. make downstream SQL choose one taxonomy explicitly

## Recommended Design

## Design Principle

Do not create a parallel `sw_*` table family unless absolutely necessary.

Instead:

- reuse existing board master/history/map tables
- load SW L1 as another industry taxonomy
- let consumers choose taxonomy with a clear filter

This minimizes DDL, avoids reworking historical-map SPs, and preserves compatibility with existing rotation code.

## Canonical Taxonomy Markers

Recommended conventions for SW L1 rows:

- `BOARD_ID`: Shenwan index code, for example `801010.SI`
- `BOARD_NAME`: Shenwan L1 industry name
- `PROVIDER`: `TUSHARE`
- `SOURCE`: `TUSHARE_SW2021_L1`

Recommended conventions for member-history rows:

- `source`: `tushare_sw_l1`

## Data Model Usage

### 1. Master table

Continue using:

- `cn_board_industry_master(BOARD_ID, BOARD_NAME, PROVIDER, ASOF_DATE, SOURCE, ...)`

Insert one snapshot per refresh date for SW L1 industry definitions.

This table is enough to store:

- taxonomy code
- taxonomy name
- source/provider
- effective snapshot date

No DDL change is strictly required.

### 2. Member history table

Continue using:

- `cn_board_industry_member_hist(board_id, symbol, valid_from, valid_to, source, ...)`

Write SW L1 constituent history here.

No DDL change is strictly required.

### 3. Daily map table

Continue using:

- `cn_board_member_map_d(trade_date, sector_type, sector_id, symbol)`

This table can already store SW L1 daily mapping because `sector_id` is generic varchar.

No DDL change is strictly required.

## Minimal DDL Recommendation

Strictly speaking, none is required for SW L1 landing.

Optional but recommended later:

1. Add a taxonomy marker to `cn_board_member_map_d`

Example:

```sql
ALTER TABLE cn_board_member_map_d
ADD COLUMN taxonomy_code VARCHAR(32) NULL;
```

Recommended values:

- `EM_INDUSTRY`
- `SW2021_L1`
- `SW2021_L3`

This is optional because current queries can already distinguish taxonomies by joining `cn_board_industry_master` or by id pattern.

For the low-cost phase, skip this DDL.

## Ingestion Plan

## Step 1. Generalize the existing Tushare loader

Current reusable script:

- [app/tools/backfill_sw_l3_history_from_tushare.py](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/app/tools/backfill_sw_l3_history_from_tushare.py)

Current limitation:

- hard-coded for `L3`
- only writes member history
- does not insert SW rows into `cn_board_industry_master`

Recommended change:

- rename to `backfill_sw_industry_history_from_tushare.py`
- support `--level L1|L2|L3`
- default to `--level L1`

### Required script changes

#### A. Replace fixed L3 code fetch with generic level fetch

Current function:

- `iter_sw_l3_codes(...)`

Change to:

- `iter_sw_codes(token: str, src: str, level: str) -> List[dict]`

The function should return enough metadata to populate master rows:

- `index_code`
- `industry_name`
- `level`
- `src`

Suggested Tushare call:

```python
ts_call(
    token=token,
    api_name="index_classify",
    params={"src": src, "level": level},
    fields="index_code,industry_name,level,industry_code,is_pub,parent_code,src",
)
```

If some fields are unavailable for the token, keep at least:

- `index_code`
- `industry_name`
- `level`
- `src`

#### B. Add master-table upsert

Before importing members, upsert SW L1 definitions into `cn_board_industry_master`.

Suggested SQL:

```sql
INSERT INTO cn_board_industry_master
    (BOARD_ID, BOARD_NAME, PROVIDER, ASOF_DATE, SOURCE, CREATED_AT, RAW_JSON)
VALUES
    (:board_id, :board_name, 'TUSHARE', :asof_date, :source, NOW(), :raw_json)
ON DUPLICATE KEY UPDATE
    BOARD_NAME = VALUES(BOARD_NAME),
    PROVIDER = VALUES(PROVIDER),
    SOURCE = VALUES(SOURCE),
    RAW_JSON = VALUES(RAW_JSON);
```

Recommended values:

- `asof_date`: script run `end` date, or a dedicated `--asof-date`
- `source`: `TUSHARE_SW2021_L1`

#### C. Keep member-history upsert unchanged except source label

Reuse the existing pattern:

```sql
INSERT INTO cn_board_industry_member_hist (board_id, symbol, valid_from, valid_to, source)
VALUES (:board_id, :symbol, :valid_from, :valid_to, :source)
ON DUPLICATE KEY UPDATE
    valid_to = VALUES(valid_to),
    source = VALUES(source),
    updated_at = CURRENT_TIMESTAMP(6);
```

Recommended source label:

- `tushare_sw_l1`

#### D. Keep map rebuild call unchanged

Continue using:

```sql
CALL sp_build_board_member_map(:d1, :d2);
```

This is one of the reasons the SW L1 landing cost stays low.

## Step 2. Rebuild daily mapping for target range

After SW L1 history import:

```sql
CALL sp_build_board_member_map('2020-01-01', '2026-04-17');
```

Or run incrementally for the latest date only:

```sql
CALL sp_build_board_member_map('2026-04-17', '2026-04-17');
```

## Step 3. Make downstream SQL choose the intended taxonomy

This is the most important SQL change.

Without this step, queries may mix:

- Eastmoney `BK%`
- Shenwan `*.SI`

## Downstream SQL / View Changes

## 1. Leader score views

Current design in:

- [docs/DDL/cn_market.cn_stock_leader_score_v1.sql](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/docs/DDL/cn_market.cn_stock_leader_score_v1.sql)
- [docs/DDL/cn_market.cn_stock_leader_score_v2.sql](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/docs/DDL/cn_market.cn_stock_leader_score_v2.sql)

Current rule:

- `sector_type = 'INDUSTRY'`
- `sector_id LIKE 'BK%'`

Recommended change:

- replace the fixed `BK%` filter with a taxonomy filter for SW L1

Recommended filter shape:

```sql
WHERE m.sector_type = 'INDUSTRY'
  AND EXISTS (
      SELECT 1
      FROM cn_board_industry_master im
      WHERE im.BOARD_ID = m.sector_id
        AND im.PROVIDER = 'TUSHARE'
        AND im.SOURCE = 'TUSHARE_SW2021_L1'
  )
```

Why this is better than `LIKE '801%'`:

- avoids hard-coding code patterns
- works if Shenwan code format changes
- keeps taxonomy choice explicit

Performance note:

- if this join becomes slow, first materialize latest master rows into a compact helper view
- do not solve this with ad-hoc full-history subqueries in every consumer

## 2. Rotation / sector aggregation chain

Likely impact files:

- [app/tools/rebuild_rotation_three_tables_from_map.py](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/app/tools/rebuild_rotation_three_tables_from_map.py)
- `docs/DDL/cn_market.cn_board_industry_eod_agg_v.sql`
- `docs/DDL/cn_market.sp_refresh_sector_eod_hist.sql`
- `docs/DDL/cn_market.cn_sector_rotation_transition_v.sql`

Recommended change:

- keep all existing table contracts unchanged
- add a taxonomy filter in the industry-name and industry-map portion
- make the chosen taxonomy explicit and stable

Recommended implementation order:

1. update `cn_board_industry_eod_agg_v` to aggregate only SW L1 industries
2. validate `cn_sector_eod_hist_t`
3. validate ranked/signal rebuild

## 3. Named views / presentation views

Any view that turns `sector_id` into a name should prefer the latest matching SW L1 row from `cn_board_industry_master`.

Recommended name lookup pattern:

```sql
SELECT x.board_id, x.board_name
FROM (
    SELECT
        m.BOARD_ID AS board_id,
        m.BOARD_NAME AS board_name,
        ROW_NUMBER() OVER (
            PARTITION BY m.BOARD_ID
            ORDER BY m.ASOF_DATE DESC
        ) AS rn
    FROM cn_board_industry_master m
    WHERE m.PROVIDER = 'TUSHARE'
      AND m.SOURCE = 'TUSHARE_SW2021_L1'
) x
WHERE x.rn = 1
```

## Suggested Script Interface

Recommended replacement CLI:

```bash
python -m app.tools.backfill_sw_industry_history_from_tushare ^
  --level L1 ^
  --src SW2021 ^
  --start 2000-01-01 ^
  --end 2026-04-17 ^
  --master-source TUSHARE_SW2021_L1 ^
  --member-source tushare_sw_l1
```

Recommended defaults:

- `--level L1`
- `--src SW2021`
- `--master-source TUSHARE_SW2021_L1`
- `--member-source tushare_sw_l1`

## Usage Notes

Implemented script:

- [app/tools/backfill_sw_industry_history_from_tushare.py](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/app/tools/backfill_sw_industry_history_from_tushare.py)

Backward-compatible wrapper:

- [app/tools/backfill_sw_l3_history_from_tushare.py](D:/LHJ/PythonWS/MarketScraper/AsharesScraperV2/app/tools/backfill_sw_l3_history_from_tushare.py)

What the new script now does:

1. pulls SW industry definitions from Tushare `index_classify`
2. upserts industry master rows into `cn_board_industry_master`
3. imports constituent history into `cn_board_industry_member_hist`
4. optionally rebuilds `cn_board_member_map_d`

Recommended SW L1 full load:

```powershell
$env:TUSHARE_TOKEN="YOUR_TOKEN"
python -m app.tools.backfill_sw_industry_history_from_tushare `
  --level L1 `
  --src SW2021 `
  --start 2000-01-01 `
  --end 2026-04-17
```

Recommended maximum practical history load for this project:

```powershell
$env:TUSHARE_TOKEN="YOUR_TOKEN"
python -m app.tools.backfill_sw_industry_history_from_tushare `
  --level L1 `
  --src SW2021 `
  --start 2000-01-04 `
  --end 2026-05-06 `
  --map-chunk-years 1 `
  --keep-existing-member-source
```

Why `2000-01-04` is the practical project floor:

- local `cn_stock_daily_price` currently starts at `2000-01-04`
- `sp_build_board_member_map` expands history only onto trading dates that exist in `cn_stock_daily_price`
- so `cn_board_member_map_d` cannot become earlier than the local price calendar

Observed source boundary checked on `2026-05-07`:

- Tushare `SW2021` `L1` membership history itself can return much earlier `in_date`
- the earliest observed `in_date` across current L1 codes was around `1990-12-10`

If you want the raw history table to preserve everything the source provides, you may load earlier:

```powershell
$env:TUSHARE_TOKEN="YOUR_TOKEN"
python -m app.tools.backfill_sw_industry_history_from_tushare `
  --level L1 `
  --src SW2021 `
  --start 1990-12-10 `
  --end 2026-05-06 `
  --map-chunk-years 1 `
  --keep-existing-member-source
```

Important interpretation:

- this earlier run can enrich `cn_board_industry_member_hist`
- but `cn_board_member_map_d` still only starts from the earliest local price date, currently `2000-01-04`

Recommended SW L1 incremental refresh:

```powershell
$env:TUSHARE_TOKEN="YOUR_TOKEN"
python -m app.tools.backfill_sw_industry_history_from_tushare `
  --level L1 `
  --src SW2021 `
  --start 2026-04-17 `
  --end 2026-04-17
```

Recommended SW L3 compatibility run:

```powershell
$env:TUSHARE_TOKEN="YOUR_TOKEN"
python -m app.tools.backfill_sw_l3_history_from_tushare --start 2000-01-01 --end 2026-04-17
```

Useful options:

- `--asof-date`
  - controls the `ASOF_DATE` written into `cn_board_industry_master`
- `--skip-master`
  - only refresh member history and map
- `--skip-map`
  - skip rebuilding `cn_board_member_map_d`
- `--keep-existing-member-source`
  - do not clear previous rows for the chosen member source before load

## Validation SQL

## 1. Check master load

```sql
SELECT PROVIDER, SOURCE, COUNT(*) AS board_cnt, MIN(ASOF_DATE) AS min_d, MAX(ASOF_DATE) AS max_d
FROM cn_board_industry_master
WHERE PROVIDER = 'TUSHARE'
  AND SOURCE = 'TUSHARE_SW2021_L1'
GROUP BY PROVIDER, SOURCE;
```

Expected:

- around 28 boards for SW L1

## 2. Check member history load

```sql
SELECT source, COUNT(*) AS row_cnt, COUNT(DISTINCT board_id) AS board_cnt
FROM cn_board_industry_member_hist
WHERE source = 'tushare_sw_l1'
GROUP BY source;
```

Expected:

- `board_cnt` around 28

## 3. Check daily map coverage

```sql
SELECT trade_date, COUNT(*) AS row_cnt, COUNT(DISTINCT sector_id) AS board_cnt
FROM cn_board_member_map_d
WHERE sector_type = 'INDUSTRY'
  AND sector_id IN (
      SELECT BOARD_ID
      FROM cn_board_industry_master
      WHERE PROVIDER = 'TUSHARE'
        AND SOURCE = 'TUSHARE_SW2021_L1'
  )
GROUP BY trade_date
ORDER BY trade_date DESC
LIMIT 20;
```

## 4. Check no taxonomy mixing in leader score

```sql
SELECT industry_id, industry_name, COUNT(*) AS cnt
FROM cn_stock_leader_score_v1
WHERE trade_date = '2026-02-27'
GROUP BY industry_id, industry_name
ORDER BY industry_id;
```

Expected after view switch:

- no `BK%`
- only SW L1 ids

## Industry Index Recommendation

SW L1 landing and SW industry-index landing should be treated as two separate phases.

### Phase A

Do first:

- stock to SW L1 industry mapping
- industry aggregation and leader grouping

### Phase B

Do later:

- load Shenwan industry indexes into `cn_index_daily_price` or a dedicated industry-index table

Reason:

- current business need is classification first
- index loading is independent and can be added later without blocking SW L1 classification

## Cost Estimate

This is a low-cost change because it reuses the current data model.

Estimated work items:

1. Generalize one existing script
2. Add one master-table upsert
3. Change a few taxonomy filters in views/SP SQL
4. Rebuild map and validate

Compared with creating a separate SW schema, this approach:

- avoids new core tables
- avoids a new map-building SP
- avoids widespread downstream refactors

## Recommended Implementation Order

1. Generalize `backfill_sw_l3_history_from_tushare.py` into level-aware SW importer
2. Load SW L1 master + member history
3. Rebuild `cn_board_member_map_d`
4. Patch leader-score SQL to use SW L1 taxonomy
5. Patch sector aggregation / rotation SQL to use SW L1 taxonomy
6. Validate counts and sample names
7. Only after that, decide whether to ingest SW industry indexes

## Final Recommendation

Adopt SW L1 by extending the current `cn_board_industry_*` pipeline, not by introducing a second parallel industry model.

This gives the best tradeoff between:

- low implementation cost
- historical compatibility
- minimal downstream breakage
- clear future path to SW industry indexes
