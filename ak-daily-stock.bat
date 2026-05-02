@echo off
setlocal

rem set "PYTHON_EXE=C:\Apps\Python\Python312\python.exe"
set "PYTHON_EXE=C:\Users\nling\AppData\Local\Python\bin\python.exe"
set "RUNNER_PY=D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2\runner.py"
set "WORKDIR=D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2"

cd /d "%WORKDIR%"

REM Daily / event-level data only.
REM stock with --flag tu uses default --days 1 smart latest-day fill.
echo [DAILY] 1/3 stock + board + stock_basic + rotation
set STOCK_BASIC_ENABLED=1
set STOCK_BASIC_PROVIDER=tushare
set STOCK_BASIC_CALENDAR_SOURCE=price
set STOCK_BASIC_LOOKBACK_DAYS=7
set STOCK_BASIC_DATE_ORDER=asc
set STOCK_BASIC_BATCH_SIZE=0
"%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks stock,board,stock_basic,rotation --asof latest --refresh
if errorlevel 1 goto :fail

echo [DAILY] 2/3 index
"%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks index --asof latest --refresh
if errorlevel 1 goto :fail

echo [DAILY] 3/3 event_daily + event_periodic smart check
echo event_daily: forecast / express / disclosure
echo event_periodic: fina/income/balancesheet smart skip unless needed
"%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks event_daily,event_periodic --asof latest --refresh
if errorlevel 1 goto :fail

echo [DAILY] done
exit /b 0

:fail
echo [DAILY] failed with exit code %ERRORLEVEL%
exit /b %ERRORLEVEL%
