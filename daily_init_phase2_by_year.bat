@echo off
setlocal EnableExtensions DisableDelayedExpansion

REM ============================================================
REM GrowthAlpha V8 daily by year runner
REM Usage:
REM   daily_by_year.bat 2010 2014 --refresh
REM   daily_by_year.bat 2010 2026 --refresh
REM ============================================================

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

if "%~1"=="" (
    echo Usage: daily_by_year.bat START_YEAR END_YEAR [--refresh] [--replace]
    popd >nul
    exit /b 2
)
if "%~2"=="" (
    echo Usage: daily_by_year.bat START_YEAR END_YEAR [--refresh] [--replace]
    popd >nul
    exit /b 2
)

set "START_YEAR=%~1"
set "END_YEAR=%~2"
shift /1
shift /1

set "PASS_ARGS="
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--refresh" (
    set "PASS_ARGS=%PASS_ARGS% --refresh"
) else if /I "%~1"=="--replace" (
    set "PASS_ARGS=%PASS_ARGS% --replace"
) else (
    echo [DAILY_BY_YEAR] WARNING: ignoring unsupported argument: %~1
)
shift /1
goto parse_args
:args_done

echo ============================================================
echo Running daily.bat year by year: %START_YEAR% to %END_YEAR%
echo Extra args:%PASS_ARGS%
echo Tasks: %V8_DAILY_TASKS%
echo ============================================================

for /L %%Y in (%START_YEAR%,1,%END_YEAR%) do (
    echo.
    echo ============================================================
    echo [DAILY] Year %%Y
    echo ============================================================
    call "%SCRIPT_DIR%daily.bat" %%Y %PASS_ARGS%
    if errorlevel 1 (
        echo.
        echo ERROR: daily.bat failed for year %%Y
        popd >nul
        exit /b 1
    )
)

echo.
echo [DAILY_BY_YEAR] completed %START_YEAR% to %END_YEAR%
popd >nul
exit /b 0
