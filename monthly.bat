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
set "EXTRA_ARGS=%EXTRA_ARGS% %~1"
shift
goto :parse_args

:after_parse_args
if "%SAW_REPLACE%"=="1" (
    echo [MONTHLY] NOTICE: --replace requested, but runner.py does not support --replace. Using --refresh/upsert behavior instead.
)
if "%SAW_REPLACE%"=="1" if "%SAW_REFRESH%"=="0" (
    set "EXTRA_ARGS=%EXTRA_ARGS% --refresh"
)
if "%EXTRA_ARGS%"=="" set "EXTRA_ARGS=--asof latest --days 31"


rem Shared Tushare transport defaults
set "TUSHARE_PRO_TIMEOUT_SECONDS=45"
set "TUSHARE_PRO_HTTP_MAX_RETRIES=4"
set "TUSHARE_PRO_HTTP_BACKOFF_SECONDS=1.5"

rem Monthly audit and auto-repair defaults
set "V8_MONTHLY_AUTO_REPAIR=1"
set "V8_MONTHLY_AUDIT_LOOKBACK_DAYS=365"
set "V8_MONTHLY_REPAIR_LOOKBACK_DAYS=60"
set "V8_MONTHLY_REPAIR_MAX_LOOKBACK_DAYS=1095"
set "V8_MONTHLY_INCLUDE_DERIVED_REFRESH=1"
set "V8_MONTHLY_INCLUDE_VALIDATIONS=1"
set "V8_MONTHLY_INCLUDE_CROSSWALK_LATEST=0"
if "%V8_ENABLE_CROSSWALK_LATEST%"=="" set "V8_ENABLE_CROSSWALK_LATEST=0"

cd /d "%WORKDIR%"

REM Monthly split responsibilities:
REM   1) v8_monthly_refresh -> financial statements/monthly basic + periodic events
REM   2) v8_monthly_audit   -> coverage audit + targeted index repair
REM   3) v8_monthly_derived -> downstream derived rebuild/validations
REM
REM Intentionally excluded:
REM   - crosswalk latest refresh, which is a V7/V8 compatibility artifact rather than
REM     a future-system primary data product

for %%T in (v8_monthly_refresh v8_monthly_audit v8_monthly_derived) do (
    echo [MONTHLY] %%T %EXTRA_ARGS%
    "%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks %%T %EXTRA_ARGS%
    if errorlevel 1 goto :fail
)

REM ---- Build static base mapping tables (not daily-updated) ----
REM cn_local_industry_map_hist records stock ↔ SW industry membership with
REM in_date/out_date. It is a static mapping table sourced from Tushare
REM index_member_all, only updated when industry classifications change.
REM Run monthly to pick up any membership changes.
echo [MONTHLY] build_local_industry_map_hist (L1)
"%PYTHON_EXE%" "%WORKDIR%\scripts\build_local_industry_map_hist.py" --start 1990-01-01 --end 2026-12-31 --level L1 --resume
if errorlevel 1 goto :fail

echo [MONTHLY] build_local_industry_map_hist (L2)
"%PYTHON_EXE%" "%WORKDIR%\scripts\build_local_industry_map_hist.py" --start 1990-01-01 --end 2026-12-31 --level L2 --resume
if errorlevel 1 goto :fail

echo [MONTHLY] build_local_industry_map_hist (L3)
"%PYTHON_EXE%" "%WORKDIR%\scripts\build_local_industry_map_hist.py" --start 1990-01-01 --end 2026-12-31 --level L3 --resume
if errorlevel 1 goto :fail

echo [MONTHLY] done
exit /b 0

:fail
echo [MONTHLY] failed with exit code %ERRORLEVEL%
exit /b %ERRORLEVEL%
