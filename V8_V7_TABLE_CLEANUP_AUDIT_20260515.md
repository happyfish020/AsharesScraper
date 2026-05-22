# V8 / V7 Table Cleanup Audit — 2026-05-15

Scope:
- Source schema dump: `Dump20260515.sql`
- Code scan root: `AshareScaperV2-20260515.zip`
- Goal: identify V7-era tables that V8 should stop using / can quarantine or drop.

## Executive conclusion

Do **not** directly drop every table containing `v7`.

There are three different classes:

1. **Drop/quarantine candidates**: V7 raw legacy data tables not referenced by AshareScraperV2 runtime code.
2. **Compatibility-backed legacy tables**: old `_v7` tables are not referenced directly by runtime code, but current non-v7 compatibility views still point to them. These cannot be dropped until the view is repointed or removed.
3. **Transition/crosswalk tables**: names contain `v7`, but they are currently V8 migration infrastructure. Keep temporarily until V8 local/SW-compatible industry path is fully independent.

## V7-named objects found in dump

| Object | Runtime code reference? | Dependency / reason | Action |
|---|---:|---|---|
| `cn_event_disclosure_date_v7` | No direct runtime ref | Backing table for view `cn_event_disclosure_date` | Do not drop until view is rebuilt or removed |
| `cn_event_earnings_forecast_v7` | No direct runtime ref | Backing table for view `cn_event_earnings_forecast` | Do not drop until view is rebuilt or removed |
| `cn_stock_leader_sw_l1_latest_snap_v7` | No direct runtime ref | Backing table for view `cn_stock_leader_sw_l1_latest_snap` | Do not drop until view is rebuilt or removed |
| `cn_sw_industry_daily_v7` | No direct runtime ref | Backing table for view `cn_sw_industry_daily` | Do not drop until view is rebuilt/repointed to V8 source |
| `cn_event_dividend_v7` | No runtime ref found | Legacy V7 event table | Quarantine/drop candidate |
| `cn_fund_etf_hist_em_v7` | No runtime ref found | Legacy V7 fund table | Quarantine/drop candidate |
| `cn_fut_index_his_v7` | No runtime ref found | Legacy V7 futures/index table | Quarantine/drop candidate |
| `cn_option_sse_daily_v7` | No runtime ref found | Legacy V7 option table | Quarantine/drop candidate |
| `cn_v7_v8_industry_crosswalk` | Yes | Used by `sw_v7_v8_crosswalk.py`, validation, V8 dataset task | Keep for now |
| `cn_v7_v8_industry_crosswalk_latest` | Yes | Used by `stock_v8_to_v7_sw_map_latest.py`, crosswalk builder | Keep for now |
| `cn_stock_v8_to_v7_sw_map_latest` | Yes | Used by V8-to-SW latest map builder | Keep for now |

## Recommended cleanup order

### Phase 0 — Safety snapshot

Before any cleanup:

```sql
CREATE DATABASE IF NOT EXISTS cn_market_legacy_v7_archive;
-- Optional: dump only legacy objects first with mysqldump before rename/drop.
```

### Phase 1 — Quarantine only, no destructive drop

Rename clear unused legacy raw tables first. This tests whether any hidden/manual process still depends on them.

Candidates:

```sql
RENAME TABLE cn_event_dividend_v7 TO zz_legacy_v7_cn_event_dividend_v7;
RENAME TABLE cn_fund_etf_hist_em_v7 TO zz_legacy_v7_cn_fund_etf_hist_em_v7;
RENAME TABLE cn_fut_index_his_v7 TO zz_legacy_v7_cn_fut_index_his_v7;
RENAME TABLE cn_option_sse_daily_v7 TO zz_legacy_v7_cn_option_sse_daily_v7;
```

Run your normal V8 daily jobs and backfill smoke after this. If nothing breaks, these can be dropped later.

### Phase 2 — Remove compatibility view dependency

These must be handled as a group because the current non-v7 names are compatibility views over `_v7` tables:

- `cn_event_disclosure_date` -> `cn_event_disclosure_date_v7`
- `cn_event_earnings_forecast` -> `cn_event_earnings_forecast_v7`
- `cn_stock_leader_sw_l1_latest_snap` -> `cn_stock_leader_sw_l1_latest_snap_v7`
- `cn_sw_industry_daily` -> `cn_sw_industry_daily_v7`

Options:

1. **If V8 still needs the logical table name**, replace the backing source with a real V8 table or V8-compatible view.
2. **If V8 does not need it**, drop the compatibility view first, then quarantine/drop the backing `_v7` table.

Do not drop the `_v7` table while the view still exists.

### Phase 3 — Keep transition tables until V8 industry replacement is accepted

Keep these for now:

- `cn_v7_v8_industry_crosswalk`
- `cn_v7_v8_industry_crosswalk_latest`
- `cn_stock_v8_to_v7_sw_map_latest`

They are not old production V7 data tables; they are migration/bridge assets currently referenced by runtime code.

Only remove them after:

1. `cn_local_industry_proxy_daily` / `cn_sw_compat_proxy_daily` have passed validation.
2. V8 no longer needs V7/SW bridge mapping for leader/industry features.
3. All scripts under `data_pipeline/builders/` and `app/tasks/v8_dataset_ops_task.py` are migrated away from these tables.

## Smoke tests after quarantine

Recommended commands/checks:

```bash
python runner.py --flag tu --tasks rotation --asof latest --days 1
python scripts/audit_cn_market_data_assets.py --db-name cn_market_red
python scripts/validate_replacement_window.py --db-name cn_market_red
```

Also run a narrow V8 smoke that touches:

- rotation tables
- mainline/lifecycle builders
- unified alpha builders
- event/fundamental loaders if enabled

## Bottom line

Immediate safe cleanup candidates are only:

```text
cn_event_dividend_v7
cn_fund_etf_hist_em_v7
cn_fut_index_his_v7
cn_option_sse_daily_v7
```

The rest either back compatibility views or are active transition/crosswalk infrastructure.
