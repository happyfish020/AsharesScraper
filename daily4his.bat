@echo off
setlocal

set SCRIPT_DIR=%~dp0
set PYTHON_EXE=python
set RUNNER_PY=%SCRIPT_DIR%runner.py

set EXTRA_ARGS=%*

for %%T in (
    v8_daily_market_raw
    v8_daily_reference
    v8_rotation_repair
    v8_daily_audit
    v8_daily_derived_foundation
    v8_daily_derived_mainline
    v8_daily_derived_alpha
) do (
    echo [DAILY] %%T %EXTRA_ARGS%
    "%PYTHON_EXE%" "%RUNNER_PY%" --flag tu --tasks %%T %EXTRA_ARGS%
    if errorlevel 1 goto :fail
)

echo Daily pipeline completed successfully.
exit /b 0

:fail
echo [DAILY] failed with exit code %ERRORLEVEL%
exit /b %ERRORLEVEL%