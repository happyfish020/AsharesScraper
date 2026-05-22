@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Stock daily price historical backfill by year
REM Purpose:
REM   Fill missing rows in cn_stock_daily_price using the existing runner CLI.
REM
REM Usage:
REM   stock_daily_price_by_year.bat
REM   stock_daily_price_by_year.bat 2010 2012
REM   stock_daily_price_by_year.bat 2010 2026 2026-05-15
REM
REM Notes:
REM   - This project CLI does NOT support: runner.py daily_price --replace
REM   - Correct task is: --tasks v8_stock
REM   - --days 9999 is intentional. Without it, StockLoaderTask may take
REM     the latest-day fast path when look_back_days == 1 and ignore the
REM     explicit historical start/end window.
REM   - --refresh is intentional. It clears scanned/failed state so a previous
REM     partial run does not skip symbols that still have historical gaps.
REM ============================================================

set SCRIPT_DIR=%~dp0
set PYTHON_EXE=python
set RUNNER_PY=%SCRIPT_DIR%runner.py

set START_YEAR=%~1
if "%START_YEAR%"=="" set START_YEAR=2010

set END_YEAR=%~2
if "%END_YEAR%"=="" set END_YEAR=2012

REM Optional third arg, useful when END_YEAR is current year.
set FINAL_END_DATE=%~3
if "%FINAL_END_DATE%"=="" set FINAL_END_DATE=

echo ============================================================
echo Running cn_stock_daily_price backfill year by year
echo Range: %START_YEAR% to %END_YEAR%
echo Task:  --tasks v8_stock
echo Table: cn_stock_daily_price
echo ============================================================

for /L %%Y in (%START_YEAR%,1,%END_YEAR%) do (
    set YEAR=%%Y
    set START_DATE=%%Y-01-01
    set END_DATE=%%Y-12-31

    if not "%FINAL_END_DATE%"=="" (
        if %%Y==%END_YEAR% set END_DATE=%FINAL_END_DATE%
    )

    echo.
    echo ============================================================
    echo [STOCK_DAILY_PRICE] Year %%Y : !START_DATE! to !END_DATE!
    echo [STOCK_DAILY_PRICE] Target table: cn_stock_daily_price
    echo ============================================================

    "%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks v8_stock --start-date !START_DATE! --end-date !END_DATE! --days 9999 --refresh
    if errorlevel 1 (
        echo.
        echo ERROR: cn_stock_daily_price backfill failed for year %%Y
        echo Command failed: "%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks v8_stock --start-date !START_DATE! --end-date !END_DATE! --days 9999 --refresh
        exit /b 1
    )
)

echo.
echo ============================================================
echo cn_stock_daily_price historical backfill completed successfully.
echo Next recommended checks:
echo   daily.bat
echo   weekly_by_year.bat %START_YEAR% %END_YEAR%
echo ============================================================

endlocal
exit /b 0
