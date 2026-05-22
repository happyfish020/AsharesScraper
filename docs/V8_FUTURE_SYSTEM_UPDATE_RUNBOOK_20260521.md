# V8 Future-System Update Runbook

Date: 2026-05-21 (updated 2026-05-21)

## Purpose

This runbook defines the current operational update contract for the future-system
path in `AsharesScraperV2`.

The goal is to keep refreshing data still used by the V8 / GrowthAlpha future
path, while stopping default refresh of legacy compatibility tables that are no
longer required in routine daily/weekly/monthly production updates.

## Primary Rule

Default production update chains should keep refreshing:

- `cn_stock_daily_price`
- `cn_index_daily_price`
- `cn_stock_daily_basic`
- `cn_sw_industry_daily`
- `cn_stock_monthly_basic`
- `cn_stock_fina_indicator`
- `cn_stock_income`
- `cn_stock_balancesheet`
- `cn_stock_cashflow`
- `cn_event_*`
- `cn_stock_fundamental_daily`
- `cn_stock_quality_score_daily`
- `cn_industry_capital_flow_daily`
- `cn_ga_stock_role_map_daily`
- `cn_stock_mainline_strength_daily`
- `cn_ga_mainline_radar_daily`
- `cn_ga_market_pulse_daily`
- **`cn_local_industry_proxy_daily`** ← added 2026-05-21 (daily derived, required by lifecycle)
- `cn_mainline_lifecycle_daily`
- `cn_unified_alpha_score_daily`
- `cn_stock_leader_sw_l1_latest_snap`

Default production update chains should not refresh legacy compatibility assets:

- `cn_board_concept_member_hist`
- `cn_board_industry_member_hist`
- `cn_board_member_map_d`
- legacy rotation snapshot tables driven by `SectorRotationSnapshotTask`
- `cn_v7_v8_industry_crosswalk_latest`

Notes:

- `cn_v7_v8_industry_crosswalk` and `cn_v7_v8_industry_crosswalk_latest` are still
  valid compatibility artifacts, but they are no longer refreshed by default in
  routine future-system ops.
- `cn_board_*` tables are kept for legacy/research/recovery workflows only.

### Static Base Tables (refreshed monthly, not daily)

These tables are **static mapping tables** that do not change on a daily basis.
They are refreshed monthly via `monthly.bat` rather than in the daily pipeline:

- `cn_local_industry_map_hist` — stock ↔ SW industry membership with in_date/out_date.
  Sourced from Tushare `index_member_all`. Only updated when industry classifications
  change (e.g., new stocks added to an industry).

## Current Batch Contract

### Daily

Entrypoint:

```bat
daily_spot_update.bat
```

Current default behavior:

- refresh market raw tables
- refresh `sw_industry`
- skip daily event loader by default
- skip rotation snapshot by default
- run derived foundation / mainline / alpha chain
- **mainline chain now includes `build_local_industry_proxy_daily.py`** (runs after market_pulse, before lifecycle)
- keep `crosswalk_latest` disabled

Default legacy exclusions:

- `V8_SKIP_ROTATION_SNAPSHOT=1`
- `V8_ENABLE_CROSSWALK_LATEST=0`
- `V8_DAILY_INCLUDE_CROSSWALK_LATEST=0`

Reason:

- `SectorRotationSnapshotTask` still depends on `cn_board_member_map_d` and legacy
  rotation tables, so it is not part of the future-system default daily refresh.

#### Mainline Chain Execution Order

The `_run_derived_mainline_chain` now executes in this order:

1. `build_cn_stock_mainline_strength_daily.py` — stock-level mainline strength
2. `build_ga_mainline_radar_daily.py` — mainline radar (role-based scoring)
3. `build_ga_market_pulse_daily.py` — market pulse indicators
4. **`build_local_industry_proxy_daily.py`** — industry proxy (member_count, ret_eqw, etc.)
5. `build_mainline_lifecycle_daily.py` — lifecycle state machine (depends on proxy data)

Step 4 is controlled by `V8_SKIP_LOCAL_INDUSTRY_PROXY` (default: 0 = run).

### Weekly

Entrypoint:

```bat
weekly.bat
```

Current default task list:

1. `v8_stock_basic`
2. `event_periodic`
3. `v8_weekly_audit_market`
4. `v8_weekly_finalize`

Current default behavior:

- refresh `cn_stock_daily_basic`
- rebuild leader-score materialization
- refresh periodic event tables
- run stock/index market audit only
- refresh latest leader SW L1 snapshot
- do not refresh board membership tables
- do not run board-reference audit
- do not refresh `crosswalk_latest`

Default legacy exclusions:

- `V8_WEEKLY_BUILD_CROSSWALK_LATEST=0`
- `V8_ENABLE_CROSSWALK_LATEST=0`
- no call to `v8_weekly_refresh`
- no call to `v8_weekly_audit`

Reason:

- `v8_weekly_refresh` bundles `BoardMembershipRefreshTask`, which writes
  `cn_board_*` legacy tables.
- `v8_weekly_audit` asserts `cn_board_concept_member_hist`,
  `cn_board_industry_member_hist`, and `cn_board_member_map_d`, so it is not
  suitable once those tables are no longer part of the routine future-system
  refresh contract.

### Monthly

Entrypoint:

```bat
monthly.bat
```

Current default behavior:

- refresh monthly/fundamental source tables
- refresh periodic event tables
- run monthly market audit
- run derived refresh chain
- keep `crosswalk_latest` disabled
- **build `cn_local_industry_map_hist`** (L1/L2/L3, after v8 tasks)

Default legacy exclusions:

- `V8_MONTHLY_INCLUDE_CROSSWALK_LATEST=0`
- `V8_ENABLE_CROSSWALK_LATEST=0`

#### Monthly Static Table Refresh

After the v8 task chain completes, `monthly.bat` runs:

```bat
python scripts/build_local_industry_map_hist.py --start 1990-01-01 --end 2026-12-31 --level L1 --resume
python scripts/build_local_industry_map_hist.py --start 1990-01-01 --end 2026-12-31 --level L2 --resume
python scripts/build_local_industry_map_hist.py --start 1990-01-01 --end 2026-12-31 --level L3 --resume
```

These refresh the stock ↔ SW industry membership mapping from Tushare
`index_member_all`. `--resume` skips already-completed industry chunks.

## Data Dependency & Refresh Strategy

### Table Classification

| Table | Type | Refresh | Pipeline | Audit Strategy |
|---|---|---|---|---|
| `cn_local_industry_map_hist` | Static mapping | Monthly | `monthly.bat` | Not audited in daily pipeline (no trade_date column) |
| `cn_local_industry_proxy_daily` | Daily derived | Daily | `daily_spot_update.bat` (mainline chain) | Audits `cn_stock_daily_price` and `cn_stock_daily_basic` only |
| `cn_mainline_lifecycle_daily` | Daily derived | Daily | `daily_spot_update.bat` (mainline chain) | Audits upstream radar/strength/flow/pulse/proxy tables |

### Dependency Flow

```
cn_stock_daily_price ─┐
                      ├──> build_local_industry_proxy_daily ──┐
cn_local_industry_map_hist ┘                                  │
                                                              ├──> build_mainline_lifecycle_daily
cn_ga_mainline_radar_daily ───────────────────────────────────┘
cn_stock_mainline_strength_daily ─────────────────────────────┘
cn_industry_capital_flow_daily ───────────────────────────────┘
cn_ga_market_pulse_daily ─────────────────────────────────────┘
```

## Legacy Recovery / Research Workflows

These are still allowed, but should be run explicitly rather than through the
default future-system batch files:

- `python runner.py --tasks board ...`
- `python runner.py --tasks v8_board_refresh ...`
- `python runner.py --tasks rotation ...`
- `python runner.py --tasks v8_rotation ...`
- `python runner.py --tasks v8_weekly_refresh ...`
- `python runner.py --tasks v8_weekly_audit ...`
- `python scripts/build_v7_v8_crosswalk_latest.py --replace`

Use those only when you intentionally need:

- board/concept history maintenance
- legacy rotation snapshot maintenance
- V7/V8 compatibility latest crosswalk outputs

## Recommended Operator Overrides

If you need to temporarily re-enable a legacy path for investigation:

```bat
set V8_SKIP_ROTATION_SNAPSHOT=0
set V8_ENABLE_CROSSWALK_LATEST=1
set V8_WEEKLY_BUILD_CROSSWALK_LATEST=1
set V8_MONTHLY_INCLUDE_CROSSWALK_LATEST=1
```

If you need to skip the industry proxy builder in the daily pipeline:

```bat
set V8_SKIP_LOCAL_INDUSTRY_PROXY=1
daily_spot_update.bat
```

If you need to restore legacy board refresh explicitly, do not use the default
`weekly.bat`. Run the board-specific tasks directly.

## Manual Backfill Commands

### cn_local_industry_map_hist (static mapping, monthly refresh)

```bat
python scripts/build_local_industry_map_hist.py --start 1990-01-01 --end 2026-12-31 --level L1 --resume
python scripts/build_local_industry_map_hist.py --start 1990-01-01 --end 2026-12-31 --level L2 --resume
python scripts/build_local_industry_map_hist.py --start 1990-01-01 --end 2026-12-31 --level L3 --resume
```

### cn_local_industry_proxy_daily (daily derived, standalone backfill)

```bat
python scripts/build_local_industry_proxy_daily.py --start 2026-05-01 --end 2026-05-21 --workers 4
```

## Related Docs

- [V8_DATASET_RUNBOOK_20260515.md](./V8_DATASET_RUNBOOK_20260515.md)
- [BK_TS_SW_ACTIVE_DEPENDENCY_AUDIT_20260521.md](./BK_TS_SW_ACTIVE_DEPENDENCY_AUDIT_20260521.md)
- [board_membership_refresh_playbook.md](./board_membership_refresh_playbook.md)
