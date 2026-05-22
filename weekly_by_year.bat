@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Weekly historical backfill by year
REM Supports both old positional usage and option-style usage:
REM   weekly_by_year.bat
REM   weekly_by_year.bat 2010 2026
REM   weekly_by_year.bat --refresh --replace --start-date 2010-01-01 --end-date 2010-12-31
REM
REM Notes:
REM   runner.py currently supports --refresh but NOT --replace.
REM   This wrapper accepts --replace for operator compatibility, but does
REM   not forward it to runner.py. It prints a notice and relies on --refresh
REM   / task-level upsert semantics.
REM ============================================================

set "DEFAULT_START_YEAR=2010"
set "DEFAULT_END_YEAR=2026"
set "DEFAULT_FINAL_END_DATE=2026-05-15"

set "START_YEAR="
set "END_YEAR="
set "START_DATE_ARG="
set "END_DATE_ARG="
set "FORWARD_ARGS="
set "SAW_REFRESH=0"
set "SAW_REPLACE=0"
set "POSITIONAL_COUNT=0"

:parse_args
if "%~1"=="" goto :after_parse

if /I "%~1"=="--start-date" (
    set "START_DATE_ARG=%~2"
    if "%~2"=="" (
        echo ERROR: --start-date requires a value.
        exit /b 2
    )
    shift
    shift
    goto :parse_args
)

if /I "%~1"=="--end-date" (
    set "END_DATE_ARG=%~2"
    if "%~2"=="" (
        echo ERROR: --end-date requires a value.
        exit /b 2
    )
    shift
    shift
    goto :parse_args
)

if /I "%~1"=="--refresh" (
    set "SAW_REFRESH=1"
    set "FORWARD_ARGS=!FORWARD_ARGS! --refresh"
    shift
    goto :parse_args
)

if /I "%~1"=="--replace" (
    set "SAW_REPLACE=1"
    REM Do not forward: runner.py does not support --replace.
    shift
    goto :parse_args
)

REM Backward-compatible positional years.
set /A POSITIONAL_COUNT+=1
if !POSITIONAL_COUNT!==1 (
    set "START_YEAR=%~1"
) else if !POSITIONAL_COUNT!==2 (
    set "END_YEAR=%~1"
) else (
    REM Forward any additional unknown positional value.
    set "FORWARD_ARGS=!FORWARD_ARGS! %~1"
)
shift
goto :parse_args

:after_parse
if not "%START_DATE_ARG%"=="" set "START_YEAR=%START_DATE_ARG:~0,4%"
if not "%END_DATE_ARG%"=="" set "END_YEAR=%END_DATE_ARG:~0,4%"

if "%START_YEAR%"=="" set "START_YEAR=%DEFAULT_START_YEAR%"
if "%END_YEAR%"=="" set "END_YEAR=%DEFAULT_END_YEAR%"

if "%SAW_REPLACE%"=="1" (
    echo [WEEKLY_BY_YEAR] NOTICE: --replace requested, but runner.py does not support --replace. Using --refresh/upsert behavior instead.
)

REM If user only passed --replace, make sure we still refresh.
if "%SAW_REPLACE%"=="1" if "%SAW_REFRESH%"=="0" (
    set "FORWARD_ARGS=!FORWARD_ARGS! --refresh"
)

echo ============================================================
echo Running weekly.bat year by year: %START_YEAR% to %END_YEAR%
echo Forward args:%FORWARD_ARGS%
echo ============================================================

for /L %%Y in (%START_YEAR%,1,%END_YEAR%) do (
    set "YEAR=%%Y"

    if %%Y==%START_YEAR% if not "%START_DATE_ARG%"=="" (
        set "START_DATE=%START_DATE_ARG%"
    ) else (
        set "START_DATE=%%Y-01-01"
    )

    if not %%Y==%START_YEAR% set "START_DATE=%%Y-01-01"

    if %%Y==%END_YEAR% (
        if not "%END_DATE_ARG%"=="" (
            set "END_DATE=%END_DATE_ARG%"
        ) else (
            if "%END_YEAR%"=="%DEFAULT_END_YEAR%" (
                set "END_DATE=%DEFAULT_FINAL_END_DATE%"
            ) else (
                set "END_DATE=%%Y-12-31"
            )
        )
    ) else (
        set "END_DATE=%%Y-12-31"
    )

    echo.
    echo ============================================================
    echo [WEEKLY] Year %%Y : !START_DATE! to !END_DATE!
    echo ============================================================

    call weekly.bat --start-date !START_DATE! --end-date !END_DATE! %FORWARD_ARGS%
    if errorlevel 1 (
        echo.
        echo ERROR: weekly.bat failed for year %%Y
        exit /b 1
    )
)

echo.
echo ============================================================
echo Weekly historical backfill completed successfully.
echo ============================================================

endlocal
exit /b 0
