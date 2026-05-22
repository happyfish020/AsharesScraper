@echo off
setlocal

REM Run strict stock+index market coverage audit only. This does not download weekly reference data.
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=python
set RUNNER_PY=%SCRIPT_DIR%runner.py

"%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks v8_weekly_audit_market %*
exit /b %ERRORLEVEL%
