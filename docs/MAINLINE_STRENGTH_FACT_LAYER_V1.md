# Mainline Strength Fact Layer V1

## Purpose

`cn_mainline_strength_fact_daily` is the clean upstream fact table for GrowthAlpha mainline strength consumption.
It is created before any replacement of `cn_ga_mainline_radar_daily`.

## Non-negotiable rule

This builder does **not** read `cn_ga_mainline_radar_daily`.
Radar may later be rebuilt from the fact table, but the fact table must never be rebuilt from radar.

## New files

- `scripts/build_cn_mainline_strength_fact_daily.py`
- `scripts/validate_cn_mainline_strength_fact_daily.py`

## Scheduler integration

`app/tasks/v8_dataset_ops_task.py` now runs the fact builder inside `_run_derived_mainline_chain()`:

1. after the first `build_cn_stock_mainline_strength_daily.py` pass
2. after the second pass and before final radar rebuild
3. validator runs when `include_validations=True`

Skip flag:

```bat
set V8_SKIP_MAINLINE_STRENGTH_FACT=1
```

## Source tables

Required:

- `cn_ga_stock_role_map_daily`
- `cn_stock_daily_price`

Optional:

- `cn_stock_daily_basic`
- `cn_industry_capital_flow_daily`

Forbidden:

- `cn_ga_mainline_radar_daily`

## Output table

`cn_mainline_strength_fact_daily`

Core fields:

- identity: `trade_date`, `mainline_id`, `mainline_name`, `source_layer`
- membership: `member_count`, `active_member_count`, `leader_count`, `core_count`
- returns/RS: `ret_1d`, `ret_5d`, `ret_20d`, `ret_60d`, `ret_120d`, `rs_20d`, `rs_60d`, `rs_120d`
- capital: `amount_total`, `amount_rank_pct`, `amount_delta_5d`, `turnover_avg`
- breadth: `up_ratio`, `strong_stock_count`, `new_high_20d_count`, `new_high_52w_count`, `breakout_count`, `breakout_ratio`
- scores: `leader_strength_score`, `breadth_score`, `capital_score`, `trend_score`, `mainline_strength_score`, `rank_no`
- quality: `data_quality_flag`, `coverage_start_date`, `is_backtest_eligible`

## Example commands

```bat
python scripts/build_cn_mainline_strength_fact_daily.py --start 2024-01-01 --end 2026-06-12 --replace --chunk-months 1
python scripts/validate_cn_mainline_strength_fact_daily.py --start 2024-01-01 --end 2026-06-12 --min-rows 1 --strict
```

## Next step after V1

Only after this table is backfilled and audited should downstream code begin replacing reads of `cn_ga_mainline_radar_daily`.
