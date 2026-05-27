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
REM Use "%~1" with quotes to prevent cmd.exe from splitting --start-date
set "EXTRA_ARGS=%EXTRA_ARGS% "%~1""
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
if "%V8_MONTHLY_INCLUDE_ALPHA%"=="" set "V8_MONTHLY_INCLUDE_ALPHA=1"
if "%V8_MONTHLY_MAP_LOOKBACK_DAYS%"=="" set "V8_MONTHLY_MAP_LOOKBACK_DAYS=365"
if "%V8_MONTHLY_FULL_MAP_HIST%"=="" set "V8_MONTHLY_FULL_MAP_HIST=0"
if "%V8_ENABLE_CROSSWALK_LATEST%"=="" set "V8_ENABLE_CROSSWALK_LATEST=0"

cd /d "%WORKDIR%"

for /f %%D in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd')"') do set "END_DATE=%%D"
for /f %%D in ('powershell -NoProfile -Command "(Get-Date).AddDays(-%V8_MONTHLY_MAP_LOOKBACK_DAYS%).ToString('yyyy-MM-dd')"') do set "MAP_START_DATE=%%D"
if "%V8_MONTHLY_FULL_MAP_HIST%"=="1" set "MAP_START_DATE=1990-01-01"
if "%V8_MONTHLY_FULL_MAP_HIST%"=="1" set "END_DATE=2026-12-31"

for %%T in (v8_monthly_refresh v8_monthly_audit v8_monthly_derived) do (
    echo [MONTHLY] %%T %EXTRA_ARGS%
    "%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks %%T %EXTRA_ARGS%
    if errorlevel 1 goto :fail
)

echo [MONTHLY] build_local_industry_map_hist L1 window %MAP_START_DATE% to %END_DATE%
"%PYTHON_EXE%" "%WORKDIR%\\scripts\\build_local_industry_map_hist.py" --start "%MAP_START_DATE%" --end "%END_DATE%" --level L1 --resume
if errorlevel 1 goto :fail

echo [MONTHLY] build_local_industry_map_hist L2 window %MAP_START_DATE% to %END_DATE%
"%PYTHON_EXE%" "%WORKDIR%\\scripts\\build_local_industry_map_hist.py" --start "%MAP_START_DATE%" --end "%END_DATE%" --level L2 --resume
if errorlevel 1 goto :fail

echo [MONTHLY] build_local_industry_map_hist L3 window %MAP_START_DATE% to %END_DATE%
"%PYTHON_EXE%" "%WORKDIR%\\scripts\\build_local_industry_map_hist.py" --start "%MAP_START_DATE%" --end "%END_DATE%" --level L3 --resume
if errorlevel 1 goto :fail

echo [MONTHLY] done
exit /b 0

:fail
echo [MONTHLY] failed with exit code %ERRORLEVEL%
exit /b %ERRORLEVEL%
