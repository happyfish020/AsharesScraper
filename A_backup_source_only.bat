@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM GrowthAlpha V7 source/docs/report/sqlite backup
REM Output: source_only_yyyyMMdd_HHmmss.zip
REM
REM Includes:
REM   - scripts\
REM   - system_layers\
REM   - reports\daily_operational_report\20260605\daily_operational_report.md
REM   - data\live_observation\growthalpha_live_observation.sqlite3
REM
REM Excludes cache/runtime/temp directories and binary cache files.
REM ============================================================

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"

set "ROOT=%CD%"
set "ZIPFILE=%ROOT%\source_only_%TS%.zip"
set "TMPDIR=%TEMP%\ga_source_backup_%TS%"
set "PS1=%TEMP%\ga_source_backup_%TS%.ps1"

 

echo.
echo Creating source backup...
echo ZIP: %ZIPFILE%
echo.

if exist "%TMPDIR%" rmdir /s /q "%TMPDIR%"
mkdir "%TMPDIR%" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Failed to create temp directory: %TMPDIR%
    exit /b 1
)

REM ============================================================
REM Copy source folders, excluding heavy/cache/runtime directories
REM ============================================================


 
copy /Y runner.py "%TMPDIR%\crunner.py"

if exist "%ROOT%\app" (
    robocopy "%ROOT%\app" "%TMPDIR%\app" /E /NFL /NDL /NJH /NJS /NP ^
        /XF *.pyc *.pyo *.pyd *.log *.tmp *.bak ^
        /XD __pycache__ .pytest_cache .mypy_cache .ruff_cache .git .idea .vscode venv .venv env reports logs cache tmp temp data downloads artifacts >nul
    if !ERRORLEVEL! GEQ 8 (
        echo [ERROR] robocopy failed while copying app. ExitCode=!ERRORLEVEL!
        if exist "%TMPDIR%" rmdir /s /q "%TMPDIR%"
        exit /b 1
    )
) else (
    echo [WARN] system_layers folder not found: %ROOT%\app
)



if exist "%ROOT%\scripts" (
    robocopy "%ROOT%\scripts" "%TMPDIR%\scripts" /E /NFL /NDL /NJH /NJS /NP ^
        /XF *.pyc *.pyo *.pyd *.log *.tmp *.bak ^
        /XD __pycache__ .pytest_cache .mypy_cache .ruff_cache .git .idea .vscode venv .venv env reports logs cache tmp temp data downloads artifacts >nul
    if !ERRORLEVEL! GEQ 8 (
        echo [ERROR] robocopy failed while copying scripts. ExitCode=!ERRORLEVEL!
        if exist "%TMPDIR%" rmdir /s /q "%TMPDIR%"
        exit /b 1
    )
) else (
    echo [WARN] scripts folder not found: %ROOT%\scripts
)

if exist "%ROOT%\data_pipeline" (
    robocopy "%ROOT%\data_pipeline" "%TMPDIR%\data_pipeline" /E /NFL /NDL /NJH /NJS /NP ^
        /XF *.pyc *.pyo *.pyd *.log *.tmp *.bak ^
        /XD __pycache__ .pytest_cache .mypy_cache .ruff_cache .git .idea .vscode venv .venv env reports logs cache tmp temp data downloads artifacts >nul
    if !ERRORLEVEL! GEQ 8 (
        echo [ERROR] robocopy failed while copying data_pipeline. ExitCode=!ERRORLEVEL!
        if exist "%TMPDIR%" rmdir /s /q "%TMPDIR%"
        exit /b 1
    )
) else (
    echo [WARN] system_layers folder not found: %ROOT%\data_pipeline
)

REM Optional docs backup. Keep disabled by default.
REM if exist "%ROOT%\docs" (
REM     robocopy "%ROOT%\docs" "%TMPDIR%\docs" /E /NFL /NDL /NJH /NJS /NP ^
REM         /XF *.pyc *.pyo *.pyd *.log *.tmp *.bak ^
REM         /XD __pycache__ .pytest_cache .mypy_cache .ruff_cache .git .idea .vscode venv .venv env reports logs cache tmp temp data downloads artifacts >nul
REM     if !ERRORLEVEL! GEQ 8 (
REM         echo [ERROR] robocopy failed while copying docs. ExitCode=!ERRORLEVEL!
REM         if exist "%TMPDIR%" rmdir /s /q "%TMPDIR%"
REM         exit /b 1
REM     )
REM )

REM ============================================================
REM Add required daily report and live-observation SQLite DB
REM These are copied explicitly because reports/data folders are excluded above.
REM ============================================================

 

REM ============================================================
REM Create a temporary PowerShell script to avoid BAT line-continuation issues
REM Keep only source/doc/report/sqlite file types before compression.
REM ============================================================

> "%PS1%" echo $ErrorActionPreference = 'Stop'
>> "%PS1%" echo $tmp = '%TMPDIR%'
>> "%PS1%" echo $zip = '%ZIPFILE%'
>> "%PS1%" echo $keep = @('.py','.sql','.yaml','.yml','.json','.ini','.cfg','.toml','.md','.txt','.rst','.bat','.ps1','.csv','.xlsx','.sqlite3')
>> "%PS1%" echo Get-ChildItem -LiteralPath $tmp -Recurse -File ^| Where-Object { $keep -notcontains $_.Extension.ToLowerInvariant() } ^| Remove-Item -Force
>> "%PS1%" echo if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
>> "%PS1%" echo $items = Get-ChildItem -LiteralPath $tmp -Force
>> "%PS1%" echo if (-not $items) { throw 'Nothing to compress. Temp directory is empty.' }
>> "%PS1%" echo Compress-Archive -LiteralPath $items.FullName -DestinationPath $zip -Force

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%"
if errorlevel 1 (
    echo.
    echo [ERROR] Backup failed during PowerShell compression.
    if exist "%PS1%" del /f /q "%PS1%"
    if exist "%TMPDIR%" rmdir /s /q "%TMPDIR%"
    exit /b 1
)

if exist "%PS1%" del /f /q "%PS1%"
if exist "%TMPDIR%" rmdir /s /q "%TMPDIR%"

echo.
echo [SUCCESS] Created:
echo %ZIPFILE%
echo.
 

endlocal
exit /b 0
