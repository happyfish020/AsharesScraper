# Leader Score Docs

This document reflects the current implemented leader-score pipeline in local MySQL.

## Status

Implemented objects:

- `cn_stock_daily_basic`
- `cn_stock_leader_score_v1`
- `cn_stock_leader_score_v2`
- `app.tools.sync_cn_stock_daily_basic_from_tushare`

Current object type in local `cn_market`:

- `cn_stock_leader_score_v2` is a `VIEW`, not a physical table

Current DB check on `2026-05-05`:

- `cn_stock_leader_score_v2` rows on `2026-04-29`: `0`
- `cn_stock_leader_score_v2` rows on `2026-04-30`: `0`

Current loaded market-cap coverage:

- `cn_stock_daily_basic`
- historical coverage depends on your latest backfill run
- total rows loaded: `163956`

Important date boundary:

- stock price data is newer than industry mapping
- leader scoring is only reliable up to the latest date in `cn_board_member_map_d`
- at the time of implementation, that latest industry-mapping date is `2026-02-27`

## Taxonomy Rule

Leader ranking currently uses only one industry taxonomy:

- `sector_type = 'INDUSTRY'`
- `sector_id LIKE 'BK%'`

This avoids mixing Eastmoney `BK%` industry IDs with Shenwan-style `%.SI` IDs in the
same ranking query.

Practical implication in the current DB:

- late-April 2026 board mapping is no longer aligned with this `BK%`-only rule
- do not treat `cn_stock_leader_score_v2` as a reliable late-April 2026 latest snapshot source

## Data Sources

`cn_stock_leader_score_v1` uses:

- `cn_board_member_map_d`
- `cn_board_industry_master`
- `cn_stock_daily_price_active_v`

`cn_stock_leader_score_v2` adds:

- `cn_stock_daily_basic`

`cn_stock_daily_basic` is loaded from Tushare `daily_basic`.

## Field Mapping

Market-cap-related fields now available in MySQL:

- `total_mv`: total market cap
- `circ_mv`: float market cap
- `total_share`
- `float_share`
- `free_share`

Other daily-basic fields loaded:

- `pe`
- `pe_ttm`
- `pb`
- `ps`
- `ps_ttm`
- `dv_ratio`
- `dv_ttm`
- `turnover_rate_f`
- `volume_ratio`

## V1 Logic

`cn_stock_leader_score_v1` is the executable fallback version using current local data only.

Layers:

- structural: not available in `v1`
- liquidity: `turnover_20d_percentile >= 0.8`
- trend: `rs_percentile >= 0.7`

Notes:

- `turnover_20d_avg` is computed from `AMOUNT`
- `rs_percentile` is computed from 20-day price return within each industry
- `leader_score_v1` ranges from `0` to `2`

## V2 Logic

`cn_stock_leader_score_v2` is the full 3-layer model.

Structural layer:

- market-cap metric: `COALESCE(total_mv, circ_mv)`
- pass when:
  - `market_cap_rank <= 3`
  - or `market_cap_percentile >= 0.9`

Liquidity layer:

- `turnover_20d_percentile >= 0.8`

Trend layer:

- `rs_percentile >= 0.7`

Final score:

```sql
leader_score =
    leader_structural
  + leader_liquidity
  + leader_trend
```

Buckets:

- `3` => `CORE_LEADER`
- `2` => `NEAR_LEADER`
- `1` => `EDGE_LEADER`
- `0` => `NON_LEADER`

## Tushare Load

Create the table and load daily-basic data with:

```powershell
python -m app.tools.sync_cn_stock_daily_basic_from_tushare --start 2026-01-09 --end 2026-02-27 --calendar-source board-map
```

Useful options:

- `--provider auto`
  - tries `tushare` first, then falls back to free `akshare`
- `--calendar-source board-map`
  - only loads dates supported by current industry mapping
- `--calendar-source price`
  - loads by stock-price trading dates
- `--skip-views`
  - loads `cn_stock_daily_basic` but skips rebuilding leader-score views

Free-source notes:

- current free fallback uses `ak.stock_individual_info_em(symbol=...)`
- available fields include `总股本 / 流通股 / 总市值 / 流通市值`
- this is snapshot-oriented, not full historical daily-basic history
- when free fallback is used, the loader writes only the latest requested date

## Runner Task

A runner task is now available:

- `stock_basic`

Example:

```powershell
python -m app.cli --tasks stock_basic --asof latest
```

Default behavior:

- intended for daily use
- defaults to `price` calendar for better trade-date coverage
- refreshes a recent rolling window with upsert semantics
- rebuilds `cn_stock_leader_score_v1`
- rebuilds `cn_stock_leader_score_v2`

Useful env vars:

- `STOCK_BASIC_ENABLED=1`
- `STOCK_BASIC_FORCE=0`
- `STOCK_BASIC_PROVIDER=tushare`
- `STOCK_BASIC_CALENDAR_SOURCE=price`
- `STOCK_BASIC_LOOKBACK_DAYS=7`
- `STOCK_BASIC_DATE_ORDER=asc`
- `STOCK_BASIC_BATCH_SIZE=0`
- `STOCK_BASIC_SOURCE_LABEL=tushare_daily_basic`
- `STOCK_BASIC_AKSHARE_WORKERS=12`
- `STOCK_BASIC_AKSHARE_TIMEOUT=15`

## Apply Order

Apply these SQL files before querying leader score:

1. `docs/DDL/cn_market.cn_stock_daily_basic.sql`
2. `docs/DDL/cn_market.cn_stock_leader_score_v1.sql`
3. `docs/DDL/cn_market.cn_stock_leader_score_v2.sql`

## Example Queries

Check daily-basic market-cap fields:

```sql
SELECT symbol, trade_date, total_mv, circ_mv
FROM cn_stock_daily_basic
WHERE trade_date = '2026-02-27'
LIMIT 20;
```

Query full 3-layer leader score:

```sql
SELECT symbol, industry_name, market_cap_rank, turnover_20d_percentile, rs_percentile, leader_score, leader_bucket
FROM cn_stock_leader_score_v2
WHERE trade_date = '2026-02-27'
ORDER BY leader_score DESC, rs_percentile DESC, turnover_20d_percentile DESC
LIMIT 50;
```

Practical stock-picking pattern:

```sql
SELECT symbol, name, industry_name, leader_score, rs_percentile
FROM cn_stock_leader_score_v2
WHERE trade_date = '2026-02-27'
  AND leader_score >= 2
ORDER BY rs_percentile DESC, turnover_20d_percentile DESC
LIMIT 10;
```

## Validation Snapshot

For `2026-02-27`, the implemented pipeline produced:

- score `3`: `162`
- score `2`: `616`
- score `1`: `1662`
- score `0`: `3016`

Rows with full 3-layer readiness on that date:

- `5456`

## Known Limitations

- `cn_stock_leader_score_v2` is a full-history view and can be slow on ad-hoc queries
- for production reads, a latest-date view or snapshot table is recommended
- do not claim "current leader" for dates later than the latest industry mapping date
