@echo off
setlocal

REM ============================================================
REM GrowthAlpha V8 monthly pipeline - split mode
REM This avoids rerunning expensive completed steps after a later failure.
REM Usage:
REM   monthly_split.bat --start-date 2010-01-01 --end-date 2026-05-15
REM Resume examples:
REM   python runner.py --flag tu --tasks v8_monthly_audit %*
REM   python runner.py --flag tu --tasks v8_monthly_derived_foundation %*
REM   python runner.py --flag tu --tasks v8_monthly_derived_mainline %*
REM   python runner.py --flag tu --tasks v8_monthly_derived_alpha %*
REM ============================================================

set EXTRA_ARGS=%*
set PYTHON_EXE=python
set RUNNER_PY=runner.py
    rem     v8_monthly_refresh
   rem rem v8_monthly_derived_foundation
    rem rem v8_monthly_derived_mainline
    v8_monthly_derived_alpha
    rem v8_monthly_audit
for %%T in (
   v8_monthly_audit
) do (
    echo [MONTHLY-SPLIT] %%T %EXTRA_ARGS%
    "%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks %%T %EXTRA_ARGS%
    if errorlevel 1 goto :fail
)

echo [MONTHLY-SPLIT] completed successfully
endlocal
exit /b 0

:fail
echo [MONTHLY-SPLIT] failed with exit code %errorlevel%
endlocal
exit /b 1
