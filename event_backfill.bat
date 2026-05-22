@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM event_backfill.bat v2
REM Independent cn_event_* historical backfill tool.
REM
REM Usage:
REM   event_backfill.bat --start-date 2010-01-01 --end-date 2026-05-15
REM   event_backfill.bat --start-date 2010-01-01 --end-date 2026-05-15 --replace
REM
REM Critical:
REM   EVENT_FORCE_FULL=1 forces EventLoaderTask to honor the requested start-date,
REM   instead of using the recent incremental window based on MAX(date)-buffer.
REM ============================================================

set START_DATE=2010-01-01
set END_DATE=latest
set REPLACE_FLAG=0
set WITH_ANNS=0

:parse
if "%~1"=="" goto after_parse

if /I "%~1"=="--start-date" (
    set START_DATE=%~2
    shift
    shift
    goto parse
)

if /I "%~1"=="--end-date" (
    set END_DATE=%~2
    shift
    shift
    goto parse
)

if /I "%~1"=="--replace" (
    set REPLACE_FLAG=1
    shift
    goto parse
)

if /I "%~1"=="--refresh" (
    shift
    goto parse
)

if /I "%~1"=="--with-anns" (
    set WITH_ANNS=1
    shift
    goto parse
)

shift
goto parse

:after_parse

echo ============================================================
echo Event Backfill Tool v2 - FORCE FULL
echo Start Date: %START_DATE%
echo End Date:   %END_DATE%
echo Replace:    %REPLACE_FLAG% ^(accepted by BAT; runner.py does not support global --replace^)
echo With Anns:  %WITH_ANNS%
echo ============================================================

REM Force EventLoaderTask to use requested full range.
set EVENT_FORCE_FULL=1
set EVENT_FULL_START=%START_DATE%
set EVENT_LOOKBACK_BUFFER_DAYS=0
set EVENT_WITH_ANNS=%WITH_ANNS%

echo [EVENT] EVENT_FORCE_FULL=%EVENT_FORCE_FULL%
echo [EVENT] EVENT_FULL_START=%EVENT_FULL_START%
echo [EVENT] EVENT_LOOKBACK_BUFFER_DAYS=%EVENT_LOOKBACK_BUFFER_DAYS%

REM Daily event backfill:
REM   cn_event_earnings_forecast
REM   cn_event_earnings_express
REM   cn_event_disclosure_date
REM   optionally cn_event_announcement_meta if --with-anns
echo.
echo [EVENT] daily event historical backfill
python runner.py --flag tu --tasks v8_event_daily --start-date %START_DATE% --end-date %END_DATE% --refresh
if errorlevel 1 (
    echo [EVENT] v8_event_daily failed with exit code %errorlevel%
    exit /b %errorlevel%
)

REM Periodic event backfill:
REM   cn_event_dividend
echo.
echo [EVENT] periodic event historical backfill
python runner.py --flag tu --tasks v8_event_periodic --start-date %START_DATE% --end-date %END_DATE% --refresh
if errorlevel 1 (
    echo [EVENT] v8_event_periodic failed with exit code %errorlevel%
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo Event backfill completed successfully.
echo ============================================================
exit /b 0
