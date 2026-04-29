@echo off
setlocal

rem set "PYTHON_EXE=C:\Apps\Python\Python312\python.exe"
set "PYTHON_EXE=C:\Users\nling\AppData\Local\Python\bin\python.exe"
set "RUNNER_PY=D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2\runner.py"
set "WORKDIR=D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2"

cd /d "%WORKDIR%"

REM Weekly maintenance / slow-changing metadata only.
echo [WEEKLY] board + stock_basic
"%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks board,stock_basic --asof latest
if errorlevel 1 goto :fail

echo [WEEKLY] done
exit /b 0

:fail
echo [WEEKLY] failed with exit code %ERRORLEVEL%
exit /b %ERRORLEVEL%
