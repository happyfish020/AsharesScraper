@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"

set "PYTHON_EXE=python"
if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
)

if "%V8_DAILY_UPDATE_LOOKBACK_DAYS%"=="" set "V8_DAILY_UPDATE_LOOKBACK_DAYS=7"
if "%V8_STOCK_SPOT_REPAIR_LOOKBACK_DAYS%"=="" set "V8_STOCK_SPOT_REPAIR_LOOKBACK_DAYS=7"
if "%V8_STOCK_SPOT_REPAIR_MISSING_THRESHOLD%"=="" set "V8_STOCK_SPOT_REPAIR_MISSING_THRESHOLD=1000"
if "%V8_STOCK_SPOT_FORCE_LATEST%"=="" set "V8_STOCK_SPOT_FORCE_LATEST=1"
if "%V8_LOCAL_FINE_MAP_REFRESH_DAYS%"=="" set "V8_LOCAL_FINE_MAP_REFRESH_DAYS=30"
if "%V8_REFRESH_LOCAL_FINE_MAP_HIST%"=="" set "V8_REFRESH_LOCAL_FINE_MAP_HIST=1"
if "%V8_DAILY_REPAIR_FULL_DERIVED%"=="" set "V8_DAILY_REPAIR_FULL_DERIVED=0"
set "V8_STOCK_DAILY_MODE=spot_repair"

set "SW_INDUSTRY_DAILY_ENABLED=0"

REM Daily is now a light operational refresh: raw market data, daily_basic,
REM leader score materialization, role/mainline/lifecycle tables.
REM Financial-quality and unified-alpha refreshes are owned by weekly/monthly jobs.
set "V8_SKIP_STOCK_FUNDAMENTAL_DAILY=1"
set "V8_SKIP_STOCK_QUALITY_SCORE=1"
set "V8_SKIP_INDUSTRY_CAPITAL_FLOW=0"
set "V8_SKIP_GA_STOCK_ROLE_MAP=0"
set "V8_SKIP_MAINLINE_STRENGTH=0"
set "V8_SKIP_MAINLINE_RADAR=0"
set "V8_SKIP_MARKET_PULSE=0"
set "V8_SKIP_LOCAL_INDUSTRY_PROXY=0"
set "V8_SKIP_MAINLINE_LIFECYCLE=0"
set "V8_SKIP_UNIFIED_ALPHA=1"

if "%V8_SKIP_EVENT_LOADER%"=="" set "V8_SKIP_EVENT_LOADER=1"
if "%V8_SKIP_ROTATION_SNAPSHOT%"=="" set "V8_SKIP_ROTATION_SNAPSHOT=1"

set "V8_DAILY_INCLUDE_VALIDATIONS=0"
set "V8_DAILY_INCLUDE_CROSSWALK_LATEST=0"
if "%V8_ENABLE_CROSSWALK_LATEST%"=="" set "V8_ENABLE_CROSSWALK_LATEST=0"

if "%V8_DAILY_TASKS%"=="" (
    set "V8_DAILY_TASKS=v8_daily_market_raw,v8_daily_reference,v8_stock_basic,v8_daily_audit,v8_daily_derived_foundation,v8_daily_derived_mainline"
)

if "%V8_SKIP_STOCK_BASIC%"=="" set "V8_SKIP_STOCK_BASIC=0"
if "%V8_SKIP_STOCK_BASIC%"=="1" (
    echo [DAILY_UPDATE] V8_SKIP_STOCK_BASIC=1, setting STOCK_BASIC_ENABLED=0
    set "STOCK_BASIC_ENABLED=0"
)

for /f %%D in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd')"') do set "END_DATE=%%D"
for /f %%D in ('powershell -NoProfile -Command "(Get-Date).AddDays(-%V8_DAILY_UPDATE_LOOKBACK_DAYS%).ToString('yyyy-MM-dd')"') do set "START_DATE=%%D"
for /f %%D in ('powershell -NoProfile -Command "(Get-Date).AddDays(-%V8_LOCAL_FINE_MAP_REFRESH_DAYS%).ToString('yyyy-MM-dd')"') do set "MAP_START_DATE=%%D"

echo ============================================================
echo [DAILY_UPDATE] Window %START_DATE% to %END_DATE%
echo [DAILY_UPDATE] mode=light-operational-refresh
echo [DAILY_UPDATE] stock_mode=%V8_STOCK_DAILY_MODE%
echo [DAILY_UPDATE] stock_repair_lookback=%V8_STOCK_SPOT_REPAIR_LOOKBACK_DAYS%
echo [DAILY_UPDATE] missing_threshold=%V8_STOCK_SPOT_REPAIR_MISSING_THRESHOLD%
echo [DAILY_UPDATE] tasks=%V8_DAILY_TASKS%
echo [DAILY_UPDATE] skip_event=%V8_SKIP_EVENT_LOADER% skip_rotation=%V8_SKIP_ROTATION_SNAPSHOT% crosswalk_enabled=%V8_ENABLE_CROSSWALK_LATEST%
echo [DAILY_UPDATE] skip_fundamental=%V8_SKIP_STOCK_FUNDAMENTAL_DAILY% skip_quality=%V8_SKIP_STOCK_QUALITY_SCORE% skip_alpha=%V8_SKIP_UNIFIED_ALPHA%
echo [DAILY_UPDATE] skip_role=%V8_SKIP_GA_STOCK_ROLE_MAP% skip_radar=%V8_SKIP_MAINLINE_RADAR% skip_proxy=%V8_SKIP_LOCAL_INDUSTRY_PROXY%
echo [DAILY_UPDATE] local_fine_map_refresh=%V8_REFRESH_LOCAL_FINE_MAP_HIST% local_fine_map_days=%V8_LOCAL_FINE_MAP_REFRESH_DAYS%
echo ============================================================

"%PYTHON_EXE%" "%SCRIPT_DIR%runner.py" --tasks "%V8_DAILY_TASKS%" --start-date "%START_DATE%" --end-date "%END_DATE%" --refresh

set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :post_success

echo [DAILY_UPDATE] First pass failed with exit code %RC%.
echo [DAILY_UPDATE] Attempting light auto-repair with expanded recent window...

set "V8_DAILY_UPDATE_LOOKBACK_DAYS=30"
for /f %%D in ('powershell -NoProfile -Command "(Get-Date).AddDays(-%V8_DAILY_UPDATE_LOOKBACK_DAYS%).ToString('yyyy-MM-dd')"') do set "REPAIR_START=%%D"
set "STOCK_BASIC_LOOKBACK_DAYS=%V8_DAILY_UPDATE_LOOKBACK_DAYS%"

if "%V8_DAILY_REPAIR_FULL_DERIVED%"=="1" (
    echo [DAILY_UPDATE] V8_DAILY_REPAIR_FULL_DERIVED=1, enabling financial-quality-alpha repair.
    set "V8_SKIP_STOCK_FUNDAMENTAL_DAILY=0"
    set "V8_SKIP_STOCK_QUALITY_SCORE=0"
    set "V8_SKIP_UNIFIED_ALPHA=0"
    set "V8_DAILY_TASKS=v8_daily_market_raw,v8_daily_reference,v8_stock_basic,v8_daily_audit,v8_daily_derived_foundation,v8_daily_derived_mainline,v8_daily_derived_alpha"
)

echo [DAILY_UPDATE] Repair window %REPAIR_START% to %END_DATE%
echo [DAILY_UPDATE] Repair tasks=%V8_DAILY_TASKS%
echo [DAILY_UPDATE] Repair stock_basic_lookback=%STOCK_BASIC_LOOKBACK_DAYS%
"%PYTHON_EXE%" "%SCRIPT_DIR%runner.py" --tasks "%V8_DAILY_TASKS%" --start-date "%REPAIR_START%" --end-date "%END_DATE%" --refresh

set "RC2=%ERRORLEVEL%"
if not "%RC2%"=="0" (
    echo [DAILY_UPDATE] Auto-repair failed with exit code %RC2%.
    exit /b %RC2%
)

echo [DAILY_UPDATE] Auto-repair completed successfully.
set "START_DATE=%REPAIR_START%"

:post_success
echo [DAILY_UPDATE] completed successfully.

if "%V8_REFRESH_LOCAL_FINE_MAP_HIST%"=="1" (
    echo [DAILY_UPDATE] Refreshing LOCAL_FINE map_hist L3 window %MAP_START_DATE% to %END_DATE% ...
    "%PYTHON_EXE%" "%SCRIPT_DIR%scripts\build_local_industry_map_hist.py" --start "%MAP_START_DATE%" --end "%END_DATE%" --level L3 --resume --workers 4
    if errorlevel 1 exit /b %errorlevel%
)

echo [DAILY_UPDATE] Refreshing board member map for %START_DATE% to %END_DATE% ...
set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"
"%PYTHON_EXE%" "%SCRIPT_DIR%app\tools\build_board_map_by_year.py" --start "%START_DATE%" --end "%END_DATE%"
if errorlevel 1 (
    echo [DAILY_UPDATE] Board map refresh had non-fatal issues, continuing...
)

echo [DAILY_UPDATE] Re-running stock_basic to refresh leader scores for %START_DATE% to %END_DATE% ...
set "STOCK_BASIC_LOOKBACK_DAYS=%V8_DAILY_UPDATE_LOOKBACK_DAYS%"
"%PYTHON_EXE%" "%SCRIPT_DIR%runner.py" --tasks v8_stock_basic --start-date "%START_DATE%" --end-date "%END_DATE%" --refresh
if errorlevel 1 (
    echo [DAILY_UPDATE] stock_basic re-run had non-fatal issues, continuing...
)

echo [DAILY_UPDATE] Re-running role_map only for %START_DATE% to %END_DATE% ...
set "V8_SKIP_STOCK_FUNDAMENTAL_DAILY=1"
set "V8_SKIP_STOCK_QUALITY_SCORE=1"
set "V8_SKIP_INDUSTRY_CAPITAL_FLOW=1"
"%PYTHON_EXE%" "%SCRIPT_DIR%runner.py" --tasks v8_daily_derived_foundation --start-date "%START_DATE%" --end-date "%END_DATE%" --refresh
if errorlevel 1 (
    echo [DAILY_UPDATE] Role map re-run had non-fatal issues, continuing...
)

echo [DAILY_UPDATE] Ensuring derived mainline tables cover full window %START_DATE% to %END_DATE% ...
"%PYTHON_EXE%" "%SCRIPT_DIR%runner.py" --tasks v8_daily_derived_mainline --start-date "%START_DATE%" --end-date "%END_DATE%" --refresh
if errorlevel 1 (
    echo [DAILY_UPDATE] Derived mainline re-run had non-fatal issues, continuing to signal guard...
)

echo [DAILY_UPDATE] Ensuring latest market/mainline signal states...
"%PYTHON_EXE%" "%SCRIPT_DIR%scripts\ensure_daily_market_mainline_signal_states.py"
if errorlevel 1 exit /b %errorlevel%

exit /b 0
