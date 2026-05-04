@echo off
setlocal

rem set "PYTHON_EXE=C:\Apps\Python\Python312\python.exe"
set "PYTHON_EXE=C:\Users\nling\AppData\Local\Python\bin\python.exe"
set "RUNNER_PY=D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2\runner.py"
set "WORKDIR=D:\LHJ\PythonWS\MarketScraper\AsharesScraperV2"

cd /d "%WORKDIR%"

REM Weekly: board membership refresh (industry/concept mapping changes infrequently).
REM stock_basic and sw_industry are now in daily.bat.
echo [WEEKLY] board membership refresh
"%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks board --asof latest
if errorlevel 1 goto :fail

echo [WEEKLY] done
exit /b 0

:fail
echo [WEEKLY] failed with exit code %ERRORLEVEL%
exit /b %ERRORLEVEL%
