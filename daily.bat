@echo off
setlocal

set "PYTHON_EXE=C:\Apps\Python\Python312\python.exe"
set "RUNNER_PY=D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2\runner.py"
set "WORKDIR=D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2"

cd /d "%WORKDIR%"

echo [DAILY] 1/3 stock + rotation
"%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks stock,rotation --asof latest
if errorlevel 1 goto :fail

echo [DAILY] 2/3 index
"%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks index --asof latest
if errorlevel 1 goto :fail

echo [DAILY] 3/3 event
"%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks event --asof latest --days 1
if errorlevel 1 goto :fail

echo [DAILY] done
exit /b 0

:fail
echo [DAILY] failed with exit code %ERRORLEVEL%
exit /b %ERRORLEVEL%
