@echo off
setlocal

rem set "PYTHON_EXE=C:\Apps\Python\Python312\python.exe"
set "PYTHON_EXE=C:\Users\nling\AppData\Local\Python\bin\python.exe"
set "RUNNER_PY=D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2\runner.py"
set "WORKDIR=D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2"

cd /d "%WORKDIR%"

REM Run on the 1st trading day of each month, after market/data sync.
REM Loads all quarterly financial data in one pass:
REM   cn_stock_monthly_basic  (PE_TTM/PB/PS month-end snapshot)
REM   cn_stock_income         (quarterly income statement)
REM   cn_stock_balancesheet   (quarterly balance sheet)
REM   cn_stock_fina_indicator (ROE / 净利润增长 / 营收增长 / OCF ratios)
REM   cn_stock_cashflow       (经营现金流, used for OCF/净利润 = ocf_to_np)
REM Then rebuilds:
REM   cn_stock_fundamental_quality_snap
REM   cn_stock_fundamental_quality_v1 (view)
REM   cn_stock_fundamental_quality_hist_v1 (view)
echo [MONTHLY] stock_fundamental (fina_indicator + cashflow + quality snap)
set "STOCK_FUNDAMENTAL_MONTHLY_FORCE=1"
"%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks stock_fundamental --asof latest
if errorlevel 1 goto :fail

echo [MONTHLY] done
exit /b 0

:fail
echo [MONTHLY] failed with exit code %ERRORLEVEL%
exit /b %ERRORLEVEL%
