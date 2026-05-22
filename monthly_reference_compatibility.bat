@echo off
setlocal

REM ============================================================
REM V8 compatibility reference monthly refresh
REM
REM Purpose:
REM   Run a wider-window monthly repair for the V8 compatibility
REM   mapping chain by reusing weekly_reference_compatibility.bat.
REM
REM Default monthly behavior:
REM   - refresh local master
REM   - replace official TS master in-scope
REM   - replace official TS member rows in-scope
REM   - rebuild crosswalk + latest snapshots
REM
REM Usage:
REM   monthly_reference_compatibility.bat [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
REM
REM   If --start-date is omitted, defaults to 50 days before today.
REM   If --end-date is omitted, defaults to 10 days before today.
REM
REM Optional env vars:
REM   V8_REFCOMP_REFRESH_LOCAL_MASTER
REM   V8_REFCOMP_REPLACE_TS_MASTER
REM   V8_REFCOMP_REPLACE_TS_MEMBERS
REM   V8_REFCOMP_SW_SRC
REM   V8_REFCOMP_TS_SRCS
REM   V8_REFCOMP_TS_LEVELS
REM   V8_REFCOMP_LOCAL_LEVEL
REM   V8_REFCOMP_WORKERS
REM   V8_REFCOMP_CHUNK_MONTHS
REM   V8_REFCOMP_OUTPUT_ROOT
REM   V8_REFCOMP_REBUILD_STOCK_MAP_LATEST
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
echo [MONTHLY-REFCOMP] WARNING: ignoring unsupported argument: %~1
shift
goto parse_args

:args_done

REM ---- Compute default dates if not provided ----
if "%START_DATE%"=="" (
    REM Default: 50 days before today
    for /f "usebackq tokens=1-3 delims=-" %%a in (`%PYTHON_EXE% -c "from datetime import date, timedelta; d=date.today()-timedelta(days=50); print(d)"`) do (
        set "START_DATE=%%a-%%b-%%c"
    )
)
if "%END_DATE%"=="" (
    REM Default: 10 days before today
    for /f "usebackq tokens=1-3 delims=-" %%a in (`%PYTHON_EXE% -c "from datetime import date, timedelta; d=date.today()-timedelta(days=10); print(d)"`) do (
        set "END_DATE=%%a-%%b-%%c"
    )
)

if "%V8_REFCOMP_REFRESH_LOCAL_MASTER%"=="" set "V8_REFCOMP_REFRESH_LOCAL_MASTER=1"
if "%V8_REFCOMP_REPLACE_TS_MASTER%"=="" set "V8_REFCOMP_REPLACE_TS_MASTER=1"
if "%V8_REFCOMP_REPLACE_TS_MEMBERS%"=="" set "V8_REFCOMP_REPLACE_TS_MEMBERS=1"
if "%V8_REFCOMP_REBUILD_STOCK_MAP_LATEST%"=="" set "V8_REFCOMP_REBUILD_STOCK_MAP_LATEST=1"

echo ============================================================
echo [MONTHLY-REFCOMP] start=%START_DATE% end=%END_DATE%
echo [MONTHLY-REFCOMP] Using monthly defaults:
echo [MONTHLY-REFCOMP]   V8_REFCOMP_REFRESH_LOCAL_MASTER=%V8_REFCOMP_REFRESH_LOCAL_MASTER%
echo [MONTHLY-REFCOMP]   V8_REFCOMP_REPLACE_TS_MASTER=%V8_REFCOMP_REPLACE_TS_MASTER%
echo [MONTHLY-REFCOMP]   V8_REFCOMP_REPLACE_TS_MEMBERS=%V8_REFCOMP_REPLACE_TS_MEMBERS%
echo [MONTHLY-REFCOMP]   V8_REFCOMP_REBUILD_STOCK_MAP_LATEST=%V8_REFCOMP_REBUILD_STOCK_MAP_LATEST%
echo ============================================================

call "%SCRIPT_DIR%weekly_reference_compatibility.bat" --start-date "%START_DATE%" --end-date "%END_DATE%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo [MONTHLY-REFCOMP] failed with exit code %EXIT_CODE%
    popd >nul
    exit /b %EXIT_CODE%
)

echo [MONTHLY-REFCOMP] completed successfully
popd >nul
exit /b 0
