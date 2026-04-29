@echo off
setlocal

rem set "PYTHON_EXE=C:\Apps\Python\Python312\python.exe"
set "PYTHON_EXE=C:\Users\nling\AppData\Local\Python\bin\python.exe"
set "RUNNER_PY=D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2\runner.py"
set "WORKDIR=D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2"

cd /d "%WORKDIR%"

REM Run on the 1st trading day of each month, after market/data sync.
REM Monthly data only: cn_stock_monthly_basic / monthly basic indicators.
echo [MONTHLY] stock_fundamental monthly data
set "STOCK_FUNDAMENTAL_MONTHLY_FORCE=1"
"%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks stock_fundamental --asof latest
if errorlevel 1 goto :fail

echo [MONTHLY] done
exit /b 0

:fail
echo [MONTHLY] failed with exit code %ERRORLEVEL%
exit /b %ERRORLEVEL%
