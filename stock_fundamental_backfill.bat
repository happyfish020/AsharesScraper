@echo off
setlocal EnableExtensions EnableDelayedExpansion

set START_DATE=2007-01-01
set END_DATE=2026-12-31
set REPLACE_FLAG=

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
    set REPLACE_FLAG=--replace
    shift
    goto parse
)
shift
goto parse

:after_parse

echo ============================================================
echo Stock Fundamental Backfill Tool
echo Start Date: %START_DATE%
echo End Date:   %END_DATE%
echo Replace:    %REPLACE_FLAG%
echo ============================================================

call runner.py --flag tu --tasks stock_fundamental.income --start-date %START_DATE% --end-date %END_DATE%
if errorlevel 1 exit /b 1

call runner.py --flag tu --tasks stock_fundamental.balancesheet --start-date %START_DATE% --end-date %END_DATE%
if errorlevel 1 exit /b 1

call runner.py --flag tu --tasks stock_fundamental.fina_indicator --start-date %START_DATE% --end-date %END_DATE%
if errorlevel 1 exit /b 1

python scripts/build_stock_fundamental_daily.py --start %START_DATE% --end %END_DATE% %REPLACE_FLAG%
if errorlevel 1 exit /b 1

echo.
echo Completed successfully.
exit /b 0
