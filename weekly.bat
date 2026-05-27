@echo off
setlocal

rem set "PYTHON_EXE=C:\Apps\Python\Python312\python.exe"
set "PYTHON_EXE=C:\Users\nling\AppData\Local\Python\bin\python.exe"
set "RUNNER_PY=D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2\runner.py"
set "WORKDIR=D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2"


REM Parse pass-through args.
REM runner.py supports --refresh but not --replace. Accept --replace at the bat layer
REM for operator compatibility, but do not forward it to runner.py.
set "EXTRA_ARGS="
set "SAW_REPLACE=0"
set "SAW_REFRESH=0"

:parse_args
if "%~1"=="" goto :after_parse_args
if /I "%~1"=="--replace" (
    set "SAW_REPLACE=1"
    shift
    goto :parse_args
)
if /I "%~1"=="--refresh" (
    set "SAW_REFRESH=1"
    set "EXTRA_ARGS=%EXTRA_ARGS% --refresh"
    shift
    goto :parse_args
)
REM Use "%~1" with quotes to prevent cmd.exe from splitting --start-date into
REM separate tokens (the hyphen is treated as a parameter delimiter by cmd.exe).
set "EXTRA_ARGS=%EXTRA_ARGS% "%~1""
shift
goto :parse_args

:after_parse_args
if "%SAW_REPLACE%"=="1" (
    echo [WEEKLY] NOTICE: --replace requested, but runner.py does not support --replace. Using --refresh/upsert behavior instead.
)
if "%SAW_REPLACE%"=="1" if "%SAW_REFRESH%"=="0" (
    set "EXTRA_ARGS=%EXTRA_ARGS% --refresh"
)
if "%EXTRA_ARGS%"=="" set "EXTRA_ARGS=--asof latest --days 7"


rem Shared Tushare transport defaults
set "TUSHARE_PRO_TIMEOUT_SECONDS=45"
set "TUSHARE_PRO_HTTP_MAX_RETRIES=4"
set "TUSHARE_PRO_HTTP_BACKOFF_SECONDS=1.5"

rem Weekly audit and auto-repair defaults
set "V8_WEEKLY_AUTO_REPAIR=1"
set "V8_WEEKLY_AUDIT_LOOKBACK_DAYS=180"
set "V8_WEEKLY_REPAIR_LOOKBACK_DAYS=30"
set "V8_WEEKLY_REPAIR_MAX_LOOKBACK_DAYS=730"
set "V8_WEEKLY_REFRESH_LATEST_SNAPSHOT=1"
set "V8_WEEKLY_BUILD_CROSSWALK_LATEST=0"
set "V8_MONTHLY_INCLUDE_ALPHA=0"
set "V8_SKIP_STOCK_FUNDAMENTAL_DAILY=0"
set "V8_SKIP_STOCK_QUALITY_SCORE=0"
set "V8_SKIP_UNIFIED_ALPHA=0"
if "%V8_ENABLE_CROSSWALK_LATEST%"=="" set "V8_ENABLE_CROSSWALK_LATEST=0"
if "%V8_WEEKLY_TASKS%"=="" set "V8_WEEKLY_TASKS=v8_stock_basic event_periodic v8_weekly_audit_market v8_monthly_derived v8_daily_derived_alpha v8_weekly_finalize"

cd /d "%WORKDIR%"

REM Weekly future-system responsibilities:
REM   1) v8_stock_basic        -> cn_stock_daily_basic + leader score materialization
REM   2) event_periodic        -> periodic event tables still used downstream
REM   3) v8_weekly_audit_market-> stock/index market coverage audit only
REM   4) v8_monthly_derived    -> weekly financial snapshot + quality-score refresh
REM   5) v8_daily_derived_alpha -> weekly unified alpha refresh after quality data is updated
REM   6) v8_weekly_finalize    -> latest leader snapshot; crosswalk latest disabled
REM
REM Intentionally excluded:
REM   - v8_weekly_refresh : bundles BoardMembershipRefreshTask and refreshes legacy cn_board_* tables
REM   - v8_weekly_audit   : asserts cn_board_* reference tables are populated
REM Daily intentionally skips stock_fundamental/quality/unified_alpha to keep trading-day maintenance light.

for %%T in (%V8_WEEKLY_TASKS%) do (
    echo [WEEKLY] %%T %EXTRA_ARGS%
    "%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks %%T %EXTRA_ARGS%
    if errorlevel 1 goto :fail
)

echo [WEEKLY] done
exit /b 0

:fail
echo [WEEKLY] failed with exit code %ERRORLEVEL%
exit /b %ERRORLEVEL%
