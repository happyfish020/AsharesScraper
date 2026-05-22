@echo off
setlocal EnableExtensions DisableDelayedExpansion

REM ============================================================
REM GrowthAlpha V8 daily runner
REM Usage:
REM   daily.bat 2014 --refresh
REM   daily.bat 2014
REM
REM This script accepts ONE year only.
REM For year ranges, use daily_by_year.bat.
REM ============================================================

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
    echo Usage: daily.bat YEAR [--refresh] [--replace]
    popd >nul
    exit /b 2
)

set "YEAR=%~1"
shift /1

set "EXTRA_ARGS="
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--refresh" (
    set "EXTRA_ARGS=%EXTRA_ARGS% --refresh"
) else if /I "%~1"=="--replace" (
    echo [DAILY] NOTICE: --replace requested, but runner.py does not support --replace. Using --refresh/upsert behavior instead.
    set "EXTRA_ARGS=%EXTRA_ARGS% --refresh"
) else (
    echo [DAILY] WARNING: ignoring unsupported argument: %~1
)
shift /1
goto parse_args
:args_done

set "START_DATE=%YEAR%-01-01"
set "END_DATE=%YEAR%-12-31"

echo [DAILY] Year %YEAR% : %START_DATE% to %END_DATE%

REM Phase-1/Phase-2 task list can be controlled externally:
REM   set V8_DAILY_TASKS=v8_daily_market_raw,v8_daily_reference,v8_daily_audit
REM   set V8_DAILY_TASKS=v8_daily_derived_foundation,v8_daily_derived_mainline,v8_daily_derived_alpha
if "%V8_DAILY_TASKS%"=="" (
    set "V8_DAILY_TASKS=v8_daily_market_raw,v8_daily_reference,v8_daily_audit,v8_daily_derived_foundation,v8_daily_derived_mainline,v8_daily_derived_alpha"
)


set V8_SKIP_ROTATION_SNAPSHOT=1
set V8_SKIP_EVENT_LOADER=1
set V8_DAILY_TASKS=v8_daily_market_raw,v8_daily_audit
rem daily_by_year.bat 2010 2014 --refresh

echo [DAILY] tasks=%V8_DAILY_TASKS%

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
