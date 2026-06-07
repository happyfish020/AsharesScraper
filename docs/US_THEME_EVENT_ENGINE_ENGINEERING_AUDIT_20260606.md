# US Theme Event Engine Engineering Audit - 2026-06-06

## Scope

This patch is based on `source_only_20260606_184327.zip` and modifies only the US/global submodule.
A-share mature collection tasks are not changed.

## Engineering gates

- Static parse: PASS (`python -m py_compile app/us_scraper/runner.py`)
- Variable-role audit: PASS
  - `update_us_risk_preference_backtest()` uses `soxx` for global-risk validation.
  - `update_us_theme_backtest()` uses `theme_proxy` for generic theme-risk validation.
  - `update_us_theme_event_engine()` consumes `us_theme_risk_backtest_daily`.
- DB table scope: PASS
  - Adds `us_theme_failure_event_daily`
  - Adds `us_theme_event_capture_daily`
  - Adds `us_theme_event_capture_summary`
- GrowthAlpha boundary: PASS
  - AshareScraper computes raw/feature/event data.
  - GrowthAlpha should consume these tables and not pull external data.

## Event definition

For each signal date and horizon 5/10/20 trading days, a theme failure event is flagged if any condition is true:

- Theme proxy max drawdown <= -8%
- Theme proxy underperforms SPY by <= -5%
- Theme proxy underperforms QQQ by <= -3.5%

## Capture definition

For each event date, the engine checks whether `theme_risk_score` crossed a threshold in the prior horizon window.
Default thresholds: 25, 40, 60.

Summary metrics include:

- capture_rate
- false_alarm_rate
- avg_lead_days
- median_lead_days
- avg_event_drawdown
- avg_captured_event_drawdown
- verdict

## Run command

```bat
python runner.py --tasks us_global --us-risk-backtest
```

## Acceptance SQL

```sql
SELECT *
FROM us_theme_event_capture_summary
ORDER BY horizon_days, warning_threshold;
```

```sql
SELECT
  horizon_days,
  warning_threshold,
  event_count,
  captured_event_count,
  capture_rate,
  false_alarm_rate,
  avg_lead_days,
  verdict
FROM us_theme_event_capture_summary
ORDER BY horizon_days, warning_threshold;
```
