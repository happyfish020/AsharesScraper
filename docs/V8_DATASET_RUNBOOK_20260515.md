# V8 Dataset Runbook

Date: 2026-05-15

## Purpose

This runbook is aligned to the current `GrowthAlpha_V7` code paths and the
current `AshareScraperV2` implementation.

`AshareScraperV2` is now responsible for both:

- raw-source synchronization
- the V8 downstream derived-table build chain used by daily production

The unified entrypoints are:

- `v8_backfill`
- `v8_daily`
- `v8_weekly`
- `v8_monthly`

For the current future-system production update contract, also see:

- [V8_FUTURE_SYSTEM_UPDATE_RUNBOOK_20260521.md](./V8_FUTURE_SYSTEM_UPDATE_RUNBOOK_20260521.md)
- [V8_LOCAL_FINE_LATEST_DAY_RECOVERY_20260525.md](./V8_LOCAL_FINE_LATEST_DAY_RECOVERY_20260525.md)

Important:

- the unified `runner.py --tasks v8_daily|v8_weekly|v8_monthly` entrypoints still exist
- but the current production batch wrappers may intentionally use a narrower task mix
  to avoid refreshing legacy compatibility tables by default

For rerun / break-fix operations, the production chains are also exposed as
independent sub-tasks:

- `v8_stock`
- `v8_index`
- `v8_board_refresh`
- `v8_stock_basic`
- `v8_sw_industry_daily`
- `v8_rotation`
- `v8_rotation_audit`
- `v8_rotation_repair`
- `v8_event_daily`
- `v8_event_periodic`
- `v8_stock_fundamental_refresh`
- `v8_daily_market_raw`
- `v8_daily_reference`
- `v8_daily_audit`
- `v8_daily_derived_foundation`
- `v8_daily_derived_mainline`
- `v8_daily_derived_alpha`
- `v8_weekly_refresh`
- `v8_weekly_audit`
- `v8_weekly_finalize`
- `v8_monthly_refresh`
- `v8_monthly_audit`
- `v8_monthly_derived`

## 1. Current V8 Operational Contract

### Raw-source tables synchronized on scraper side

- `cn_stock_daily_price`
- `cn_index_daily_price`
- `cn_stock_daily_basic`
- `cn_stock_monthly_basic`
- `cn_stock_fina_indicator`
- `cn_stock_income`
- `cn_stock_balancesheet`
- `cn_sw_industry_daily`
- `cn_board_industry_member_hist`
- `cn_board_member_map_d`
- `cn_local_industry_master`
- `cn_local_industry_map_hist`
- `cn_local_industry_proxy_daily`
- `cn_event_disclosure_date`
- `cn_event_earnings_forecast`

### Derived tables now built on scraper side

- `cn_stock_fundamental_daily`
- `cn_stock_quality_score_daily`
- `cn_industry_capital_flow_daily`
- `cn_ga_stock_role_map_daily`
- `cn_stock_mainline_strength_daily`
- `cn_ga_mainline_radar_daily`
- `cn_ga_market_pulse_daily`
- `cn_mainline_lifecycle_daily`
- `cn_unified_alpha_score_daily`
- `cn_stock_leader_sw_l1_latest_snap`
- `cn_v7_v8_industry_crosswalk_latest`

### Optional research / acceptance validation hook

- `validate_unified_alpha_leader_recall.py`

This hook is disabled by default. It is intended for research acceptance,
not for hard production freshness gating.

### Industry semantics for V8

V8 production should be interpreted using the semantics contract in
[V8_LOCAL_INDUSTRY_SEMANTICS_CONTRACT_20260525.md](./V8_LOCAL_INDUSTRY_SEMANTICS_CONTRACT_20260525.md).

Short version:

- `SW_L1` means official Shenwan 2021 level-1 industries (31 industries)
- the fine-grained V8 production industry set currently has 391 industries
- today that fine-grained set is physically stored as:
  - `cn_local_industry_map_hist.industry_level = 'L3'`
  - `cn_local_industry_proxy_daily.industry_level = 'L1'`
- that `proxy_daily.L1` label is legacy storage only; semantically it is the
  V8 `LOCAL_FINE` production layer, not official SW level-1

Operational rule:

- for V8 mainline production, treat `cn_local_industry_proxy_daily` as the
  primary daily industry source
- treat `cn_local_industry_map_hist` as the membership source used to build it
- do not require official `sw_daily` as the sole primary industry source under
  the current 2000-point Tushare operating constraint
- `scripts/build_local_industry_map_hist.py` now defaults to `--level L3`
  because `L3` is the current V8 `LOCAL_FINE` production layer

## 2. Unified Task Entry Points

### Daily

```bash
python runner.py --flag tu --tasks v8_daily --asof latest --days 1
```

Current production wrapper:

```bat
daily_spot_update.bat
```

Operator note:

- after the daily task chain succeeds, `daily_spot_update.bat` runs
  a recent-window `cn_local_industry_map_hist --level L3` refresh and then runs
  `scripts/ensure_daily_market_mainline_signal_states.py`
- that guard now resolves the latest **buildable** mainline-signal date, not
  simply the latest raw stock/index trade date
- in practice it is bounded by the minimum of:
  - latest `cn_stock_daily_price`
  - latest `cn_index_daily_price`
  - latest `cn_stock_daily_basic`
  - latest `cn_local_industry_map_hist` `LOCAL_FINE` horizon
    (`industry_level = 'L3'`)
- this avoids false failures when raw market tables are ahead of
  `cn_stock_daily_basic` or the static `LOCAL_FINE` membership refresh

Default daily chain:

1. raw refresh
2. coverage audit
3. derived build chain
4. derived validations
5. auto-repair retry if first pass fails or raw coverage has gaps

Daily sub-task rerun entrypoints:

- `v8_daily_market_raw`
  - `stock` + `index`
- `v8_daily_reference`
  - `sw_industry` + optional `rotation` + optional `event_daily`
- `v8_daily_audit`
  - coverage audit + targeted index repair from audit results
- `v8_daily_derived_foundation`
  - fundamental / quality / latest snap / capital flow / role map
- `v8_daily_derived_mainline`
  - mainline strength / radar / pulse / lifecycle + related validations
- `v8_daily_derived_alpha`
  - unified alpha + validation + crosswalk + optional leader recall

Raw refresh includes:

- `stock`
- `index`
- `sw_industry`
- optional `rotation`
- optional `event_daily`

Default daily production setting:

- `board_refresh` is not part of `v8_daily`
- `stock_basic` is not part of `v8_daily`
- `rotation` is skipped by default in the current future-system batch wrapper
- `crosswalk_latest` is disabled by default in the current future-system batch wrapper
- heavy concept / board membership maintenance and weekly reference refresh
  are left to weekly or monthly runs

Derived build chain includes:

- `build_stock_fundamental_daily.py`
- `build_stock_quality_score_daily.py`
- `build_cn_stock_leader_sw_l1_latest_snap`
- `build_industry_capital_flow_daily.py`
- `build_ga_stock_role_map_daily.py`
- `build_cn_stock_mainline_strength_daily.py` pass 1
- `build_ga_mainline_radar_daily.py` pass 1
- `build_ga_market_pulse_daily.py` pass 1
- `build_mainline_lifecycle_daily.py`
- `validate_mainline_lifecycle_daily.py`
- `build_cn_stock_mainline_strength_daily.py` pass 2
- `validate_cn_mainline_strength_daily.py`
- `build_ga_mainline_radar_daily.py` pass 2
- `build_ga_market_pulse_daily.py` pass 2
- `build_unified_alpha_score_daily.py`
- `validate_unified_alpha_score_daily.py`
- optional `build_v7_v8_crosswalk_latest.py` when explicitly enabled and the source table exists

Why `cn_mainline_lifecycle_daily` is in the middle:

- it is a real production dependency
- it feeds the later `cn_stock_mainline_strength_daily` / unified alpha chain
- its upstream radar and pulse inputs are refreshed before validation

Self-healing behavior:

- raw coverage audit checks `cn_stock_daily_price` and `cn_index_daily_price`
- daily audit window is wider than the execution window by default
- `daily.bat` still runs with `--days 1`, but audit can inspect recent history
  and trigger broader repair
- if raw coverage has gaps, or if any builder/validator in first pass fails,
  `v8_daily` retries an expanded window from the earliest detected missing date
  inside the audit horizon
- second pass reruns both raw loaders and the derived chain
- if second pass still fails, task exits with error

Default audit / repair controls:

- `V8_DAILY_AUDIT_LOOKBACK_DAYS=90`
- `V8_DAILY_REPAIR_LOOKBACK_DAYS=15`
- `V8_DAILY_REPAIR_MAX_LOOKBACK_DAYS=365`

Disable auto-repair:

- `V8_DAILY_AUTO_REPAIR=0`

### Weekly

```bash
python runner.py --flag tu --tasks v8_weekly --asof latest --days 7
```

Current production wrapper:

```bat
weekly.bat
```

Current future-system weekly chain:

- `v8_stock_basic`
- `event_periodic`
- `v8_weekly_audit_market`
- latest leader snapshot refresh
- `crosswalk_latest` disabled by default

Weekly pure-frequency responsibility:

- `v8_stock_basic`
- `v8_event_periodic`
- `v8_weekly_audit_market`
- `v8_weekly_finalize`

Weekly sub-task rerun entrypoints:

- `v8_weekly_refresh`
  - legacy bundle: `board` + `stock_basic` + `event_periodic`
- `v8_weekly_audit`
  - legacy board-reference audit
- `v8_weekly_audit_market`
  - wider stock/index market coverage audit + targeted index repair
- `v8_weekly_finalize`
  - latest leader snapshot; latest crosswalk only when explicitly enabled

Weekly self-healing behavior:

- weekly audit inspects a wider lookback window by default
- if raw coverage gaps are found, weekly task auto-expands the repair window
- second pass reruns weekly refresh tasks and audit
- if gaps still remain, task exits with error

Current future-system wrapper notes:

- default `weekly.bat` no longer calls `v8_weekly_refresh`
- default `weekly.bat` no longer calls `v8_weekly_audit`
- this avoids routine refresh of `cn_board_*` legacy tables

Default weekly controls:

- `V8_WEEKLY_AUTO_REPAIR=1`
- `V8_WEEKLY_AUDIT_LOOKBACK_DAYS=180`
- `V8_WEEKLY_REPAIR_LOOKBACK_DAYS=30`
- `V8_WEEKLY_REPAIR_MAX_LOOKBACK_DAYS=730`

### Monthly

```bash
python runner.py --flag tu --tasks v8_monthly --asof latest
```

Current production wrapper:

```bat
monthly.bat
```

Default monthly chain:

- `stock_fundamental`
- `event_periodic`
- derived refresh chain for the configured window
- optional structural refresh tasks remain suitable here as well

Monthly pure-frequency responsibility:

- `v8_stock_fundamental_refresh`
- `v8_event_periodic`
- `v8_monthly_audit`
- `v8_monthly_derived`

Monthly sub-task rerun entrypoints:

- `v8_monthly_refresh`
  - `stock_fundamental` + `event_periodic`
- `v8_monthly_audit`
  - wider coverage audit + targeted index repair
- `v8_monthly_derived`
  - full derived rebuild chain for the requested window

Current future-system wrapper notes:

- `crosswalk_latest` is disabled by default
- monthly source refresh remains active because fundamentals and derived chains are still part of the primary path

Monthly self-healing behavior:

- monthly audit also checks a broader historical window
- if coverage gaps are found, monthly task auto-expands the repair window
- second pass reruns monthly refresh and derived rebuild
- if gaps still remain, task exits with error

Behavior:

- monthly fundamental sync is forced even if the scheduler date is not the
  usual gate day
- downstream derived rebuild is enabled by default

Default monthly controls:

- `V8_MONTHLY_AUTO_REPAIR=1`
- `V8_MONTHLY_AUDIT_LOOKBACK_DAYS=365`
- `V8_MONTHLY_REPAIR_LOOKBACK_DAYS=60`
- `V8_MONTHLY_REPAIR_MAX_LOOKBACK_DAYS=1095`

## 3. Historical Backfill

### Standard full backfill

```bash
set V8_BACKFILL_START=20100101
set V8_BACKFILL_END=20260515
python runner.py --flag tu --tasks v8_backfill --asof 20260515
```

Default `v8_backfill` behavior:

- price history via `his_stocks`
- board refresh window
- `cn_stock_daily_basic` historical backfill
- monthly basic + quarterly raw fundamental sync
- `cn_sw_industry_daily` full history sync
- optional SW membership history
- optional concept history
- full derived build chain
- derived validations

Default switches:

- `V8_BACKFILL_INCLUDE_PRICE=1`
- `V8_BACKFILL_INCLUDE_FUNDAMENTAL=1`
- `V8_BACKFILL_INCLUDE_DAILY_BASIC=1`
- `V8_BACKFILL_INCLUDE_SW_DAILY=1`
- `V8_BACKFILL_INCLUDE_BOARD_REFRESH=1`
- `V8_BACKFILL_INCLUDE_DERIVED=1`
- `V8_BACKFILL_INCLUDE_VALIDATIONS=1`
- `V8_BACKFILL_INCLUDE_SW_HISTORY=0`
- `V8_BACKFILL_INCLUDE_CONCEPT_HISTORY=0`

### Price history only

```bash
set V8_BACKFILL_START=20100101
set V8_BACKFILL_END=20260515
set V8_BACKFILL_INCLUDE_PRICE=1
set V8_BACKFILL_INCLUDE_FUNDAMENTAL=0
set V8_BACKFILL_INCLUDE_DAILY_BASIC=0
set V8_BACKFILL_INCLUDE_SW_DAILY=0
set V8_BACKFILL_INCLUDE_DERIVED=0
python runner.py --flag tu --tasks v8_backfill --asof 20260515
```

### Raw fundamental + daily basic + SW daily only

```bash
set V8_BACKFILL_START=20100101
set V8_BACKFILL_END=20260515
set V8_BACKFILL_INCLUDE_PRICE=0
set V8_BACKFILL_INCLUDE_FUNDAMENTAL=1
set V8_BACKFILL_INCLUDE_DAILY_BASIC=1
set V8_BACKFILL_INCLUDE_SW_DAILY=1
set V8_BACKFILL_INCLUDE_DERIVED=0
python runner.py --flag tu --tasks v8_backfill --asof 20260515
```

### Include SW membership history

```bash
set V8_BACKFILL_START=19901210
set V8_BACKFILL_END=20260515
set V8_BACKFILL_INCLUDE_SW_HISTORY=1
set V8_BACKFILL_SW_SRC=SW2021
set V8_BACKFILL_SW_LEVEL=L1
set V8_BACKFILL_KEEP_EXISTING_MEMBER_SOURCE=1
python runner.py --flag tu --tasks v8_backfill --asof 20260515
```

This calls:

- `python -m app.tools.backfill_sw_industry_history_from_tushare`

### Include concept membership history

```bash
set V8_BACKFILL_START=20000104
set V8_BACKFILL_END=20260515
set V8_BACKFILL_INCLUDE_CONCEPT_HISTORY=1
set V8_BACKFILL_CONCEPT_SOURCE_LABEL=tushare_concept
python runner.py --flag tu --tasks v8_backfill --asof 20260515
```

This calls:

- `python -m app.tools.backfill_concept_history_from_tushare`

## 4. Recommended First-Time Build Order

### Step 1: price history

```bash
set V8_BACKFILL_START=20100101
set V8_BACKFILL_END=20260515
set V8_BACKFILL_INCLUDE_PRICE=1
set V8_BACKFILL_INCLUDE_FUNDAMENTAL=0
set V8_BACKFILL_INCLUDE_DAILY_BASIC=0
set V8_BACKFILL_INCLUDE_SW_DAILY=0
set V8_BACKFILL_INCLUDE_DERIVED=0
python runner.py --flag tu --tasks v8_backfill --asof 20260515
```

### Step 2: daily basic + raw fundamentals + SW daily

```bash
set V8_BACKFILL_START=20100101
set V8_BACKFILL_END=20260515
set V8_BACKFILL_INCLUDE_PRICE=0
set V8_BACKFILL_INCLUDE_FUNDAMENTAL=1
set V8_BACKFILL_INCLUDE_DAILY_BASIC=1
set V8_BACKFILL_INCLUDE_SW_DAILY=1
set V8_BACKFILL_INCLUDE_DERIVED=0
python runner.py --flag tu --tasks v8_backfill --asof 20260515
```

### Step 3: optional compatibility history

```bash
set V8_BACKFILL_START=19901210
set V8_BACKFILL_END=20260515
set V8_BACKFILL_INCLUDE_SW_HISTORY=1
set V8_BACKFILL_INCLUDE_CONCEPT_HISTORY=1
python runner.py --flag tu --tasks v8_backfill --asof 20260515
```

### Step 4: derived build chain

```bash
set V8_BACKFILL_START=20100101
set V8_BACKFILL_END=20260515
set V8_BACKFILL_INCLUDE_PRICE=0
set V8_BACKFILL_INCLUDE_FUNDAMENTAL=0
set V8_BACKFILL_INCLUDE_DAILY_BASIC=0
set V8_BACKFILL_INCLUDE_SW_DAILY=0
set V8_BACKFILL_INCLUDE_DERIVED=1
python runner.py --flag tu --tasks v8_backfill --asof 20260515
```

### Step 5: begin daily operations

```bash
python runner.py --flag tu --tasks v8_daily --asof latest --days 1
```

## 5. Key Environment Switches

### Daily

- `V8_DAILY_AUTO_REPAIR`
- `V8_DAILY_REPAIR_LOOKBACK_DAYS`
- `V8_DAILY_REPAIR_MAX_LOOKBACK_DAYS`
- `V8_DAILY_INCLUDE_BOARD_REFRESH`
- `V8_DAILY_INCLUDE_VALIDATIONS`
- `V8_DAILY_INCLUDE_CROSSWALK_LATEST`

### Weekly

- `V8_WEEKLY_AUTO_REPAIR`
- `V8_WEEKLY_AUDIT_LOOKBACK_DAYS`
- `V8_WEEKLY_REPAIR_LOOKBACK_DAYS`
- `V8_WEEKLY_REPAIR_MAX_LOOKBACK_DAYS`

### Monthly

- `V8_MONTHLY_AUTO_REPAIR`
- `V8_MONTHLY_AUDIT_LOOKBACK_DAYS`
- `V8_MONTHLY_REPAIR_LOOKBACK_DAYS`
- `V8_MONTHLY_REPAIR_MAX_LOOKBACK_DAYS`
- `V8_MONTHLY_INCLUDE_DERIVED_REFRESH`
- `V8_MONTHLY_INCLUDE_VALIDATIONS`
- `V8_MONTHLY_INCLUDE_CROSSWALK_LATEST`

### Backfill

- `V8_BACKFILL_INCLUDE_PRICE`
- `V8_BACKFILL_INCLUDE_FUNDAMENTAL`
- `V8_BACKFILL_INCLUDE_DAILY_BASIC`
- `V8_BACKFILL_INCLUDE_SW_DAILY`
- `V8_BACKFILL_INCLUDE_BOARD_REFRESH`
- `V8_BACKFILL_INCLUDE_SW_HISTORY`
- `V8_BACKFILL_INCLUDE_CONCEPT_HISTORY`
- `V8_BACKFILL_INCLUDE_DERIVED`
- `V8_BACKFILL_INCLUDE_VALIDATIONS`
- `V8_BACKFILL_INCLUDE_CROSSWALK_LATEST`
- `V8_BACKFILL_KEEP_EXISTING_MEMBER_SOURCE`

### Optional research validation

- `V8_ENABLE_LEADER_RECALL_VALIDATION=1`
- `V8_GROWTHALPHA_V7_ROOT=<absolute path to GrowthAlpha_V7>`
- `V8_LEADER_RECALL_CONFIG=<yaml path, absolute or relative to GrowthAlpha_V7 root>`

This hook is intentionally opt-in because it measures signal recall quality,
not table freshness.

## 6. Batch Entrypoints

The repo batch files point to the unified task names:

- `daily.bat` -> `v8_daily`
- `weekly.bat` -> `v8_weekly`
- `monthly.bat` -> `v8_monthly`

## 7. What Still Stays Outside Hard Freshness Gating

The following are not default blockers for production freshness:

- leader recall acceptance research
- custom backtest validations
- strategy-level report generation in `GrowthAlpha_V7`

They can be attached as optional downstream checks, but they should not be
confused with core data production readiness.

## 8. Suggested Scheduler Mapping

### Every trading day after close

```bash
python runner.py --flag tu --tasks v8_daily --asof latest --days 1
```

### Every weekend

```bash
python runner.py --flag tu --tasks v8_weekly --asof latest --days 7
```

### First trading day of each month

```bash
python runner.py --flag tu --tasks v8_monthly --asof latest
```

## 9. Summary

Use these four task names as the standard operational contract:

- `v8_backfill`
- `v8_daily`
- `v8_weekly`
- `v8_monthly`

`AshareScraperV2` now owns both the underlying data synchronization and the
main V8 production derived-table chain, including `cn_mainline_lifecycle_daily`
and the unified alpha daily build path.
Raw-function rerun entrypoints shared across daily / weekly / monthly:

- `v8_stock`
  - stock daily price refresh
- `v8_index`
  - index daily price refresh
- `v8_board_refresh`
  - board / concept / industry membership refresh
- `v8_stock_basic`
  - stock basic weekly refresh
- `v8_sw_industry_daily`
  - SW industry daily refresh
- `v8_rotation`
  - sector rotation snapshot refresh
- `v8_rotation_audit`
  - audit rotation upstream / BT / snap gaps
- `v8_rotation_repair`
  - repair rotation upstream / BT / snap gaps
- `v8_event_daily`
  - daily event refresh
- `v8_event_periodic`
  - periodic event refresh
- `v8_stock_fundamental_refresh`
  - monthly raw fundamental refresh
