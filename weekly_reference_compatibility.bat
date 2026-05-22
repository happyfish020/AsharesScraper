@echo off
setlocal

REM ============================================================
REM V8 compatibility reference weekly refresh
REM
REM Purpose:
REM   Refresh the V8 compatibility mapping chain that is NOT fully
REM   covered by the standard v8_daily / v8_weekly / v8_monthly tasks.
REM
REM Covered tables:
REM   - cn_local_industry_master           (optional, low frequency)
REM   - cn_local_industry_map_hist
REM   - cn_local_industry_proxy_daily
REM   - cn_ts_sw_industry_master
REM   - cn_ts_sw_industry_member_hist
REM   - cn_v7_v8_industry_crosswalk
REM   - cn_v7_v8_industry_crosswalk_latest
REM   - cn_stock_v8_to_v7_sw_map_latest
REM
REM Usage:
REM   weekly_reference_compatibility.bat [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
REM
REM   If --start-date is omitted, defaults to 30 days before today.
REM   If --end-date is omitted, defaults to 7 days before today.
REM
REM Optional env vars:
REM   V8_REFCOMP_SW_SRC=SW2021
REM   V8_REFCOMP_LOCAL_LEVEL=L1
REM   V8_REFCOMP_TS_SRCS=SW2021 SW2014
REM   V8_REFCOMP_TS_LEVELS=L1
REM   V8_REFCOMP_WORKERS=4
REM   V8_REFCOMP_CHUNK_MONTHS=1
REM   V8_REFCOMP_REFRESH_LOCAL_MASTER=0
REM   V8_REFCOMP_REPLACE_TS_MASTER=0
REM   V8_REFCOMP_REPLACE_TS_MEMBERS=1
REM   V8_REFCOMP_OUTPUT_ROOT=reports\analysis
REM   V8_REFCOMP_REBUILD_STOCK_MAP_LATEST=1
REM ============================================================

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "PYTHON_EXE=python"
if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
)

set "START_DATE="
set "END_DATE="

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--start-date" (
    set "START_DATE=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--end-date" (
    set "END_DATE=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--refresh" (
    REM Accepted for operator compatibility; individual builders are upsert/resume based.
    shift
    goto parse_args
)
if /I "%~1"=="--replace" (
    REM Accepted for operator compatibility; replace behavior is controlled per-builder below.
    shift
    goto parse_args
)
echo [REFCOMP] WARNING: ignoring unsupported argument: %~1
shift
goto parse_args

:args_done

REM ---- Compute default dates if not provided ----
if "%START_DATE%"=="" (
    REM Default: 30 days before today
    for /f "usebackq tokens=1-3 delims=-" %%a in (`%PYTHON_EXE% -c "from datetime import date, timedelta; d=date.today()-timedelta(days=30); print(d)"`) do (
        set "START_DATE=%%a-%%b-%%c"
    )
)
if "%END_DATE%"=="" (
    REM Default: 7 days before today
    for /f "usebackq tokens=1-3 delims=-" %%a in (`%PYTHON_EXE% -c "from datetime import date, timedelta; d=date.today()-timedelta(days=7); print(d)"`) do (
        set "END_DATE=%%a-%%b-%%c"
    )
)

if "%V8_REFCOMP_SW_SRC%"=="" set "V8_REFCOMP_SW_SRC=SW2021"
if "%V8_REFCOMP_LOCAL_LEVEL%"=="" set "V8_REFCOMP_LOCAL_LEVEL=L1"
if "%V8_REFCOMP_TS_SRCS%"=="" set "V8_REFCOMP_TS_SRCS=SW2021 SW2014"
if "%V8_REFCOMP_TS_LEVELS%"=="" set "V8_REFCOMP_TS_LEVELS=L1"
if "%V8_REFCOMP_WORKERS%"=="" set "V8_REFCOMP_WORKERS=4"
if "%V8_REFCOMP_CHUNK_MONTHS%"=="" set "V8_REFCOMP_CHUNK_MONTHS=1"
if "%V8_REFCOMP_REFRESH_LOCAL_MASTER%"=="" set "V8_REFCOMP_REFRESH_LOCAL_MASTER=0"
if "%V8_REFCOMP_REPLACE_TS_MASTER%"=="" set "V8_REFCOMP_REPLACE_TS_MASTER=0"
if "%V8_REFCOMP_REPLACE_TS_MEMBERS%"=="" set "V8_REFCOMP_REPLACE_TS_MEMBERS=1"
if "%V8_REFCOMP_OUTPUT_ROOT%"=="" set "V8_REFCOMP_OUTPUT_ROOT=reports\analysis"
if "%V8_REFCOMP_REBUILD_STOCK_MAP_LATEST%"=="" set "V8_REFCOMP_REBUILD_STOCK_MAP_LATEST=1"

set "TS_MASTER_FLAG="
if "%V8_REFCOMP_REPLACE_TS_MASTER%"=="1" set "TS_MASTER_FLAG=--replace-master"

set "TS_MEMBER_FLAG="
if "%V8_REFCOMP_REPLACE_TS_MEMBERS%"=="1" set "TS_MEMBER_FLAG=--replace-members"

echo ============================================================
echo [REFCOMP] start=%START_DATE% end=%END_DATE%
echo [REFCOMP] sw_src=%V8_REFCOMP_SW_SRC% local_level=%V8_REFCOMP_LOCAL_LEVEL%
echo [REFCOMP] ts_srcs=%V8_REFCOMP_TS_SRCS% ts_levels=%V8_REFCOMP_TS_LEVELS%
echo ============================================================

if "%V8_REFCOMP_REFRESH_LOCAL_MASTER%"=="1" (
    echo [REFCOMP] build_sw_industry_master.py
    "%PYTHON_EXE%" ".\scripts\build_sw_industry_master.py" --start "%START_DATE%" --end "%END_DATE%" --src %V8_REFCOMP_SW_SRC%
    if errorlevel 1 goto :fail
)

echo [REFCOMP] build_local_industry_map_hist.py
"%PYTHON_EXE%" ".\scripts\build_local_industry_map_hist.py" --start "%START_DATE%" --end "%END_DATE%" --level %V8_REFCOMP_LOCAL_LEVEL% --src %V8_REFCOMP_SW_SRC% --resume --workers %V8_REFCOMP_WORKERS%
if errorlevel 1 goto :fail

echo [REFCOMP] build_local_industry_proxy_daily.py
"%PYTHON_EXE%" ".\scripts\build_local_industry_proxy_daily.py" --start "%START_DATE%" --end "%END_DATE%" --resume --workers %V8_REFCOMP_WORKERS% --chunk-months %V8_REFCOMP_CHUNK_MONTHS% --industry-level %V8_REFCOMP_LOCAL_LEVEL%
if errorlevel 1 goto :fail

echo [REFCOMP] build_tushare_sw_replacement_sources.py
"%PYTHON_EXE%" ".\scripts\build_tushare_sw_replacement_sources.py" --start "%START_DATE%" --end "%END_DATE%" --srcs %V8_REFCOMP_TS_SRCS% --levels %V8_REFCOMP_TS_LEVELS% %TS_MASTER_FLAG% %TS_MEMBER_FLAG% --output-dir "%V8_REFCOMP_OUTPUT_ROOT%\tushare_sw_replacement_sources_weekly"
if errorlevel 1 goto :fail

echo [REFCOMP] build_v7_v8_industry_crosswalk.py
"%PYTHON_EXE%" ".\scripts\build_v7_v8_industry_crosswalk.py" --start "%START_DATE%" --end "%END_DATE%" --replace --srcs %V8_REFCOMP_TS_SRCS% --source-mode db --output-dir "%V8_REFCOMP_OUTPUT_ROOT%\sw_v7_v8_crosswalk_weekly"
if errorlevel 1 goto :fail

echo [REFCOMP] build_v7_v8_crosswalk_latest.py
"%PYTHON_EXE%" ".\scripts\build_v7_v8_crosswalk_latest.py" --replace --output-dir "%V8_REFCOMP_OUTPUT_ROOT%\v7_v8_crosswalk_latest"
if errorlevel 1 goto :fail

if "%V8_REFCOMP_REBUILD_STOCK_MAP_LATEST%"=="1" (
    echo [REFCOMP] build_stock_v8_to_v7_sw_map_latest.py
    "%PYTHON_EXE%" ".\scripts\build_stock_v8_to_v7_sw_map_latest.py" --replace --output-dir "%V8_REFCOMP_OUTPUT_ROOT%\stock_v8_to_v7_sw_map_latest"
    if errorlevel 1 goto :fail
)

echo [REFCOMP] completed successfully
popd >nul
exit /b 0

:fail
echo [REFCOMP] failed with exit code %ERRORLEVEL%
popd >nul
exit /b %ERRORLEVEL%
