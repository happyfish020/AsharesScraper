@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM daily_weekly_by_year.bat
REM For each year:
REM   1) Run daily.bat
REM   2) Run weekly_by_year.bat
REM
REM Supports:
REM   daily_weekly_by_year.bat
REM   daily_weekly_by_year.bat 2011 2026
REM   daily_weekly_by_year.bat --start-date 2011-01-01 --end-date 2026-05-15
REM   daily_weekly_by_year.bat --start-date 2011-01-01 --end-date 2026-05-15 --refresh --replace
REM
REM Notes:
REM   - --refresh is always applied.
REM   - --replace is passed only to weekly_by_year.bat because your wrapper can strip it
REM     before runner.py if runner.py does not support --replace.
REM ============================================================

set START_YEAR=
set END_YEAR=
set START_DATE_ARG=
set END_DATE_ARG=
set EXTRA_WEEKLY_ARGS=

REM Defaults
set DEFAULT_START_YEAR=2011
set DEFAULT_END_YEAR=2026
set DEFAULT_END_MMDD=12-31

REM ------------------------------------------------------------
REM Parse args
REM ------------------------------------------------------------
:parse_args
if "%~1"=="" goto after_parse

if /I "%~1"=="--start-date" (
    set START_DATE_ARG=%~2
    shift
    shift
    goto parse_args
)

if /I "%~1"=="--end-date" (
    set END_DATE_ARG=%~2
    shift
    shift
    goto parse_args
)

if /I "%~1"=="--refresh" (
    shift
    goto parse_args
)

if /I "%~1"=="--replace" (
    set EXTRA_WEEKLY_ARGS=!EXTRA_WEEKLY_ARGS! --replace
    shift
    goto parse_args
)

REM Positional year mode: first two numeric args
if "%START_YEAR%"=="" (
    set START_YEAR=%~1
    shift
    goto parse_args
)

if "%END_YEAR%"=="" (
    set END_YEAR=%~1
    shift
    goto parse_args
)

REM Ignore unknown extra args safely
shift
goto parse_args

:after_parse

REM ------------------------------------------------------------
REM Resolve years from --start-date / --end-date when provided
REM ------------------------------------------------------------
if not "%START_DATE_ARG%"=="" (
    set START_YEAR=%START_DATE_ARG:~0,4%
)

if not "%END_DATE_ARG%"=="" (
    set END_YEAR=%END_DATE_ARG:~0,4%
)

if "%START_YEAR%"=="" set START_YEAR=%DEFAULT_START_YEAR%
if "%END_YEAR%"=="" set END_YEAR=%DEFAULT_END_YEAR%


REM ============================================================
REM Safe boundary guard for derived mainline validations.
REM Default safe start date is 2011-01-01.
REM If requested start-date is earlier, clamp automatically.
REM ============================================================
set MIN_SAFE_START_DATE=2011-01-01

if not "%START_DATE_ARG%"=="" (
    if "%START_DATE_ARG%" LSS "%MIN_SAFE_START_DATE%" (
        echo [BOUNDARY GUARD] Requested start-date %START_DATE_ARG% is before safe boundary %MIN_SAFE_START_DATE%
        echo [BOUNDARY GUARD] Auto-clamping start-date to %MIN_SAFE_START_DATE%
        set START_DATE_ARG=%MIN_SAFE_START_DATE%
        set START_YEAR=2011
    )
)

if "%START_DATE_ARG%"=="" (
    if not "%START_YEAR%"=="" (
        if %START_YEAR% LSS 2011 (
            echo [BOUNDARY GUARD] Requested start year %START_YEAR% is before safe boundary 2011
            echo [BOUNDARY GUARD] Auto-clamping start year to 2011
            set START_YEAR=2011
            set START_DATE_ARG=2011-01-01
        )
    )
)


echo ============================================================
echo Running daily + weekly year by year: %START_YEAR% to %END_YEAR%
echo Requested start-date: %START_DATE_ARG%
echo Requested end-date:   %END_DATE_ARG%
echo ============================================================

for /L %%Y in (%START_YEAR%,1,%END_YEAR%) do (
    set Y=%%Y
    set LOOP_START_DATE=!Y!-01-01
    set LOOP_END_DATE=!Y!-12-31

    REM Clamp first year start date to requested --start-date.
    if not "%START_DATE_ARG%"=="" (
        if "!Y!"=="%START_YEAR%" set LOOP_START_DATE=%START_DATE_ARG%
    )

    REM Clamp final year end date to requested --end-date.
    if not "%END_DATE_ARG%"=="" (
        if "!Y!"=="%END_YEAR%" set LOOP_END_DATE=%END_DATE_ARG%
    )

    echo.
    echo ============================================================
    echo [STEP 1/2] DAILY  Year !Y! : !LOOP_START_DATE! to !LOOP_END_DATE!
    echo ============================================================
    call daily.bat --start-date !LOOP_START_DATE! --end-date !LOOP_END_DATE! --refresh
    if errorlevel 1 (
        echo ERROR: daily.bat failed for year !Y!
        exit /b 1
    )

    echo.
    echo ============================================================
    echo [STEP 2/2] WEEKLY Year !Y! : !LOOP_START_DATE! to !LOOP_END_DATE!
    echo ============================================================
    call weekly_by_year.bat --start-date !LOOP_START_DATE! --end-date !LOOP_END_DATE! --refresh !EXTRA_WEEKLY_ARGS!
    if errorlevel 1 (
        echo ERROR: weekly_by_year.bat failed for year !Y!
        exit /b 1
    )
)

echo.
echo ============================================================
echo Completed successfully: %START_YEAR% to %END_YEAR%
echo ============================================================
exit /b 0
