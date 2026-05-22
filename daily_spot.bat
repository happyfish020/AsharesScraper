@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=python"
if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
)
if "%~1"=="" (
    echo Usage: daily.bat YEAR [--refresh]
    exit /b 2
)
set "YEAR=%~1"
shift /1
if "%V8_DAILY_TASKS%"=="" (
    set "V8_DAILY_TASKS=v8_daily_market_raw,v8_daily_reference,v8_daily_audit,v8_daily_derived_foundation,v8_daily_derived_mainline,v8_daily_derived_alpha"
)
"%PYTHON_EXE%" "%SCRIPT_DIR%runner.py" --tasks "%V8_DAILY_TASKS%" --start-date "%YEAR%-01-01" --end-date "%YEAR%-12-31" %*
exit /b %ERRORLEVEL%
