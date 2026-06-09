@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"

set "PYTHON_EXE=python"
if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
)

REM Keep US/global refresh as a separate first step and fail fast if it fails.
"%PYTHON_EXE%" "%SCRIPT_DIR%runner.py" --tasks us_global
if errorlevel 1 exit /b %errorlevel%

if "%V8_DAILY_UPDATE_LOOKBACK_DAYS%"=="" set "V8_DAILY_UPDATE_LOOKBACK_DAYS=7"
if "%V8_STOCK_SPOT_REPAIR_LOOKBACK_DAYS%"=="" set "V8_STOCK_SPOT_REPAIR_LOOKBACK_DAYS=7"
if "%V8_STOCK_SPOT_REPAIR_MISSING_THRESHOLD%"=="" set "V8_STOCK_SPOT_REPAIR_MISSING_THRESHOLD=1000"
if "%V8_STOCK_SPOT_FORCE_LATEST%"=="" set "V8_STOCK_SPOT_FORCE_LATEST=1"
if "%V8_LOCAL_FINE_MAP_REFRESH_DAYS%"=="" set "V8_LOCAL_FINE_MAP_REFRESH_DAYS=30"
if "%V8_REFRESH_LOCAL_FINE_MAP_HIST%"=="" set "V8_REFRESH_LOCAL_FINE_MAP_HIST=1"
if "%V8_DAILY_REPAIR_FULL_DERIVED%"=="" set "V8_DAILY_REPAIR_FULL_DERIVED=0"
set "V8_STOCK_DAILY_MODE=spot_repair"

set "SW_INDUSTRY_DAILY_ENABLED=0"

REM Daily is split into two phases to avoid duplicate derived execution:
REM   Phase 1: raw/reference/audit only.
REM   Phase 2: dependency refresh, then stock_basic/foundation/mainline exactly once.
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

REM Do not include v8_stock_basic / v8_daily_derived_foundation / v8_daily_derived_mainline here.
REM They run once after board/map dependencies are refreshed.
if "%V8_DAILY_TASKS%"=="" (
    set "V8_DAILY_TASKS=v8_daily_market_raw,v8_daily_reference,v8_daily_audit"
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
echo [DAILY_UPDATE] mode=light-operational-refresh-no-duplicate-derived
echo [DAILY_UPDATE] stock_mode=%V8_STOCK_DAILY_MODE%
echo [DAILY_UPDATE] stock_repair_lookback=%V8_STOCK_SPOT_REPAIR_LOOKBACK_DAYS%
echo [DAILY_UPDATE] missing_threshold=%V8_STOCK_SPOT_REPAIR_MISSING_THRESHOLD%
echo [DAILY_UPDATE] phase1_tasks=%V8_DAILY_TASKS%
echo [DAILY_UPDATE] phase2_tasks=v8_stock_basic,v8_daily_derived_foundation,v8_daily_derived_mainline
echo [DAILY_UPDATE] skip_event=%V8_SKIP_EVENT_LOADER% skip_rotation=%V8_SKIP_ROTATION_SNAPSHOT% crosswalk_enabled=%V8_ENABLE_CROSSWALK_LATEST%
echo [DAILY_UPDATE] skip_fundamental=%V8_SKIP_STOCK_FUNDAMENTAL_DAILY% skip_quality=%V8_SKIP_STOCK_QUALITY_SCORE% skip_alpha=%V8_SKIP_UNIFIED_ALPHA%
echo [DAILY_UPDATE] skip_role=%V8_SKIP_GA_STOCK_ROLE_MAP% skip_radar=%V8_SKIP_MAINLINE_RADAR% skip_proxy=%V8_SKIP_LOCAL_INDUSTRY_PROXY%
echo [DAILY_UPDATE] local_fine_map_refresh=%V8_REFRESH_LOCAL_FINE_MAP_HIST% local_fine_map_days=%V8_LOCAL_FINE_MAP_REFRESH_DAYS%
echo ============================================================

echo [DAILY_UPDATE] Phase 1/2: raw/reference/audit refresh...
"%PYTHON_EXE%" "%SCRIPT_DIR%runner.py" --tasks "%V8_DAILY_TASKS%" --start-date "%START_DATE%" --end-date "%END_DATE%" --refresh

set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :post_success

echo [DAILY_UPDATE] Phase 1 failed with exit code %RC%.
echo [DAILY_UPDATE] Attempting light auto-repair with expanded recent window...

set "V8_DAILY_UPDATE_LOOKBACK_DAYS=30"
for /f %%D in ('powershell -NoProfile -Command "(Get-Date).AddDays(-%V8_DAILY_UPDATE_LOOKBACK_DAYS%).ToString('yyyy-MM-dd')"') do set "REPAIR_START=%%D"
set "STOCK_BASIC_LOOKBACK_DAYS=%V8_DAILY_UPDATE_LOOKBACK_DAYS%"

if "%V8_DAILY_REPAIR_FULL_DERIVED%"=="1" (
    echo [DAILY_UPDATE] V8_DAILY_REPAIR_FULL_DERIVED=1 requested, but daily repair still keeps derived in Phase 2 to avoid duplicate execution.
    set "V8_SKIP_STOCK_FUNDAMENTAL_DAILY=0"
    set "V8_SKIP_STOCK_QUALITY_SCORE=0"
    set "V8_SKIP_UNIFIED_ALPHA=0"
)

echo [DAILY_UPDATE] Repair window %REPAIR_START% to %END_DATE%
echo [DAILY_UPDATE] Repair phase1_tasks=%V8_DAILY_TASKS%
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
echo [DAILY_UPDATE] Phase 1 completed successfully.

set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"

echo [DAILY_UPDATE] Refreshing board member map for %START_DATE% to %END_DATE% ...
"%PYTHON_EXE%" "%SCRIPT_DIR%app\tools\build_board_map_by_year.py" --start "%START_DATE%" --end "%END_DATE%"
if errorlevel 1 (
    echo [DAILY_UPDATE] Board map refresh had non-fatal issues, continuing...
)

if "%V8_REFRESH_LOCAL_FINE_MAP_HIST%"=="1" (
    echo [DAILY_UPDATE] Refreshing LOCAL_FINE map_hist L3 window %MAP_START_DATE% to %END_DATE% ...
    "%PYTHON_EXE%" "%SCRIPT_DIR%scripts\build_local_industry_map_hist.py" --start "%MAP_START_DATE%" --end "%END_DATE%" --level L3 --resume --workers 4
    if errorlevel 1 exit /b %errorlevel%
)

if not "%V8_SKIP_STOCK_BASIC%"=="1" (
    echo [DAILY_UPDATE] Phase 2/2: running stock_basic once for %START_DATE% to %END_DATE% ...
    set "STOCK_BASIC_LOOKBACK_DAYS=%V8_DAILY_UPDATE_LOOKBACK_DAYS%"
    "%PYTHON_EXE%" "%SCRIPT_DIR%runner.py" --tasks v8_stock_basic --start-date "%START_DATE%" --end-date "%END_DATE%" --refresh
    if errorlevel 1 exit /b %errorlevel%
) else (
    echo [DAILY_UPDATE] Phase 2/2: skipped stock_basic by V8_SKIP_STOCK_BASIC=1
)

echo [DAILY_UPDATE] Phase 2/2: running derived foundation once for %START_DATE% to %END_DATE% ...
set "V8_SKIP_STOCK_FUNDAMENTAL_DAILY=1"
set "V8_SKIP_STOCK_QUALITY_SCORE=1"
"%PYTHON_EXE%" "%SCRIPT_DIR%runner.py" --tasks v8_daily_derived_foundation --start-date "%START_DATE%" --end-date "%END_DATE%" --refresh
if errorlevel 1 exit /b %errorlevel%

echo [DAILY_UPDATE] Phase 2/2: running derived mainline once for %START_DATE% to %END_DATE% ...
"%PYTHON_EXE%" "%SCRIPT_DIR%runner.py" --tasks v8_daily_derived_mainline --start-date "%START_DATE%" --end-date "%END_DATE%" --refresh
if errorlevel 1 exit /b %errorlevel%

echo [DAILY_UPDATE] Ensuring latest market/mainline signal states...
"%PYTHON_EXE%" "%SCRIPT_DIR%scripts\ensure_daily_market_mainline_signal_states.py"
if errorlevel 1 exit /b %errorlevel%

cd /d D:\LHJ\PythonWS\MarketMon\GrowthAlpha_V7
python D:\LHJ\PythonWS\MarketMon\GrowthAlpha_V7\scripts\run_daily_operational_report.py
if errorlevel 1 exit /b %errorlevel%

exit /b 0
