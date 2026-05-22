@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "PYTHON_EXE=python"
if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
)

set "RUNNER_PY=%SCRIPT_DIR%runner.py"
if not exist "%RUNNER_PY%" (
    echo [DAILY] ERROR: runner.py not found at "%RUNNER_PY%"
    popd >nul
    exit /b 1
)

if "%~1"=="" (
    echo Usage: daily_backfill.bat [YEAR ^| --start-date YYYY-MM-DD --end-date YYYY-MM-DD] [--refresh] [--replace]
    echo.
    echo Examples:
    echo   daily_backfill.bat 2026
    echo   daily_backfill.bat --start-date 2026-05-01 --end-date 2026-05-15
    popd >nul
    exit /b 2
)

set "YEAR="
set "START_DATE="
set "END_DATE="
set "EXTRA_ARGS="

:parse_args
if "%~1"=="" goto args_done

if /I "%~1"=="--start-date" (
    shift /1
    goto parse_start_date
)

if /I "%~1"=="--end-date" (
    shift /1
    goto parse_end_date
)

if /I "%~1"=="--refresh" (
    set "EXTRA_ARGS=%EXTRA_ARGS% --refresh"
    shift /1
    goto parse_args
)

if /I "%~1"=="--replace" (
    echo [DAILY] NOTICE: --replace requested, but runner.py does not support --replace. Using --refresh.
    set "EXTRA_ARGS=%EXTRA_ARGS% --refresh"
    shift /1
    goto parse_args
)

rem If not a flag, treat as YEAR (positional argument)
if not defined YEAR (
    set "YEAR=%~1"
    shift /1
    goto parse_args
)

echo [DAILY] WARNING: ignoring unsupported argument: %~1
shift /1
goto parse_args

:parse_start_date
if "%~1"=="" (
    echo [DAILY] ERROR: --start-date requires a value YYYY-MM-DD
    popd >nul
    exit /b 2
)
set "START_DATE=%~1"
shift /1
goto parse_args

:parse_end_date
if "%~1"=="" (
    echo [DAILY] ERROR: --end-date requires a value YYYY-MM-DD
    popd >nul
    exit /b 2
)
set "END_DATE=%~1"
shift /1
goto parse_args

:args_done

rem If --start-date or --end-date provided, use them directly
if defined START_DATE (
    if not defined END_DATE (
        echo [DAILY] ERROR: --start-date requires --end-date
        popd >nul
        exit /b 2
    )
) else if defined END_DATE (
    echo [DAILY] ERROR: --end-date requires --start-date
    popd >nul
    exit /b 2
) else if defined YEAR (
    rem Fallback: use YEAR to derive full-year range
    set "START_DATE=%YEAR%-01-01"
    set "END_DATE=%YEAR%-12-31"
) else (
    echo [DAILY] ERROR: provide either YEAR or --start-date/--end-date
    popd >nul
    exit /b 2
)

if "%V8_DAILY_TASKS%"=="" (
    REM v8_stock_basic loads cn_stock_daily_basic and refreshes cn_stock_leader_score_daily VIEW,
    REM which are required by derived foundation/mainline builders (build_ga_stock_role_map_daily,
    REM build_cn_stock_mainline_strength_daily, etc.). Without it, those builders find no leader
    REM score data for recent dates and silently skip, leaving derived tables empty.
    set "V8_DAILY_TASKS=v8_daily_market_raw,v8_daily_reference,v8_stock_basic,v8_daily_audit,v8_daily_derived_foundation,v8_daily_derived_mainline,v8_daily_derived_alpha"
)

echo ============================================================
echo [DAILY] Year %YEAR% : %START_DATE% to %END_DATE%
echo [DAILY] tasks=%V8_DAILY_TASKS%
echo ============================================================

"%PYTHON_EXE%" "%RUNNER_PY%" --tasks "%V8_DAILY_TASKS%" --start-date "%START_DATE%" --end-date "%END_DATE%" %EXTRA_ARGS%

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo [DAILY] failed with exit code %EXIT_CODE%
    popd >nul
    exit /b %EXIT_CODE%
)

echo [DAILY] completed year %YEAR%
popd >nul
exit /b 0
