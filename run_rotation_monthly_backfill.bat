@echo off
setlocal

REM Sequential monthly backfill:
REM 2000-01 .. 2026-02, one command per month.

python -m app.tools.run_rotation_monthly_backfill ^
  --start-ym 2000-01 ^
  --end-ym 2026-02 ^
  --top-pct 0.30 ^
  --breadth-min 0.60 ^
  --rank-signal-mode bulk_sql ^
  --months-per-chunk 1 ^
  --clear-first 1 ^
  --retries 8 ^
  --retry-sleep-sec 3

set RC=%ERRORLEVEL%
echo Exit code: %RC%
exit /b %RC%

