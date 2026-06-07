@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ==========================================================
REM GrowthAlpha cn_market Recent Mirror Sync
REM Source : cn_market_red  (PRD live)
REM Target : cn_market      (recent backup / hot mirror)
REM Mode   : schema full sync + selected data sync
REM Notes  :
REM   - No dump file is written to disk.
REM   - Every command is checked; any failure exits immediately.
REM   - Use root/admin user for DDL/import to avoid VIEW/ROUTINE permission issues.
REM ==========================================================

set MYSQL_HOST=127.0.0.1
set MYSQL_PORT=3306
set MYSQL_USER=root
set MYSQL_PWD=YOUR_PASSWORD_HERE

set SOURCE_DB=cn_market_red
set TARGET_DB=cn_market
set RECENT_DAYS=365

REM MySQL client options
set MYSQL_BASE=-h %MYSQL_HOST% -P %MYSQL_PORT% -u %MYSQL_USER% --default-character-set=utf8mb4
set DUMP_BASE=-h %MYSQL_HOST% -P %MYSQL_PORT% -u %MYSQL_USER% --default-character-set=utf8mb4 --single-transaction --set-gtid-purged=OFF --no-tablespaces

REM ==========================================================
REM FULL small/dimension/config tables
REM Add/remove table names here as needed.
REM ==========================================================
set FULL_TABLES=^
 cn_trade_calendar ^
 cn_mainline_registry ^
 cn_board_concept_master ^
 cn_board_concept_cons_current ^
 cn_board_industry_master ^
 cn_local_industry_master ^
 cn_local_task_status ^
 cn_fundamental_quality_param_t ^
 cn_baseline_registry_t ^
 cn_sector_rot_baseline_t

REM ==========================================================
REM Recent daily tables with trade_date lowercase
REM ==========================================================
set RECENT_TRADE_DATE_TABLES=^
 cn_stock_daily_basic ^
 cn_stock_fundamental_daily ^
 cn_stock_leader_score_daily ^
 cn_stock_quality_score_daily ^
 cn_unified_alpha_score_daily ^
 cn_ga_stock_role_map_daily ^
 cn_ga_mainline_radar_daily ^
 cn_ga_market_context_daily ^
 cn_ga_market_pulse_daily ^
 cn_industry_alpha_feature_daily ^
 cn_industry_alpha_score_daily ^
 cn_industry_capital_flow_daily ^
 cn_local_industry_proxy_daily ^
 cn_mainline_lifecycle_daily ^
 cn_mainline_risk_temperature_daily ^
 cn_mainline_strength_daily ^
 cn_market_breadth_daily ^
 cn_market_context_mainline_strength_daily ^
 cn_market_risk_temperature_daily ^
 cn_market_structure_daily ^
 cn_a_share_overlay_daily_alerts ^
 cn_a_share_overlay_daily_shortlist ^
 cn_a_share_overlay_daily_snapshot ^
 cn_a_share_overlay_daily_state ^
 cn_p64m_signal_analog_intelligence_daily ^
 cn_p64n_rolling_signal_calibration_daily ^
 cn_p64o_regime_signal_probability_daily ^
 cn_p64p_multilayer_context_calibration_daily ^
 cn_p64r_archetype_calibration_daily ^
 cn_p64r_archetype_signal_quality_daily ^
 cn_p64r_context_archetype_daily ^
 cn_p64s_probability_fallback_daily ^
 cn_p64t_signal_quality_governance_daily ^
 cn_p64u_operational_signal_priority_daily ^
 cn_p65_decision_audit_daily ^
 cn_p65_live_signal_observation_daily ^
 cn_p65_operational_observation_daily ^
 cn_p65_signal_failure_audit_daily ^
 cn_p70_cognitive_compression_daily ^
 cn_p70_cognitive_conflict_daily ^
 cn_p70_operational_reality_daily ^
 cn_p70_unified_cognition_daily ^
 cn_p73_stage2_adapter_daily ^
 cn_p73_stage2_asset_registry_daily ^
 cn_p74_rc1_operational_consolidation_daily ^
 cn_p74_rc2_dynamic_narrative_daily ^
 cn_p74_rc3_real_runtime_consumption_daily ^
 cn_p74_rc4_artifact_runtime_daily ^
 cn_p74_rc5_operational_narrative_daily ^
 cn_p74_rc5_runtime_graph_daily ^
 cn_p74_rc5_runtime_graph_source_daily ^
 cn_p74_real_operational_consolidation_daily ^
 cn_p75a_rc1_runtime_rebuild_daily ^
 cn_p75a_rc2_runtime_execution_daily ^
 cn_p75a_rc3_runtime_verification_daily ^
 cn_p75a_runtime_population_daily ^
 cn_p76_runtime_governance_daily ^
 cn_p77a_rc3_daily_report_governance ^
 cn_s1a_holding_state_daily ^
 cn_s1b_hold_quality_daily ^
 cn_s1c_risk_stress_daily ^
 cn_s1d_mainline_alignment_daily ^
 cn_s1e_reentry_watch_daily ^
 cn_s1f_operational_review_daily ^
 cn_s1g_operational_briefing_daily ^
 cn_s2a_exit_event_daily ^
 cn_s2b_exit_reason_daily ^
 cn_s2c_exit_confidence_daily ^
 cn_s2d_reentry_allowed_daily ^
 cn_s2e_exit_narrative_daily ^
 cn_s2f_exit_review_daily ^
 cn_s2g_exit_operational_briefing_daily ^
 cn_s3a_reentry_candidate_daily ^
 cn_s3b_repair_confirmation_daily ^
 cn_s3c_stress_normalization_daily ^
 cn_s3d_mainline_survival_daily ^
 cn_s3e_reentry_quality_score_daily ^
 cn_s3f_reentry_workspace_daily ^
 cn_s3g_reentry_operational_briefing_daily ^
 cn_s4a_mainline_state_snapshot_daily ^
 cn_s4b_lifecycle_transition_daily ^
 cn_s4c_mainline_health_score_daily ^
 cn_s4d_lifecycle_risk_monitor_daily ^
 cn_s4e_action_context_daily ^
 cn_s4f_mainline_lifecycle_workspace_daily ^
 cn_s4g_mainline_lifecycle_briefing_daily ^
 cn_s5a_shadow_candidate_daily ^
 cn_s5b_shadow_structure_score_daily ^
 cn_s5c_shadow_mainline_fit_daily ^
 cn_s5d_shadow_risk_filter_daily ^
 cn_s5e_shadow_watch_priority_daily ^
 cn_s5f_shadow_discovery_workspace_daily ^
 cn_s5g_shadow_discovery_briefing_daily

REM ==========================================================
REM Recent daily tables with TRADE_DATE uppercase
REM ==========================================================
set RECENT_TRADE_DATE_UPPER_TABLES=^
 cn_stock_daily_price ^
 cn_index_daily_price ^
 cn_fut_index_his ^
 cn_sector_energy_snap_t ^
 cn_sector_rot_bt_daily_t ^
 cn_sector_rot_pos_daily_t ^
 cn_rotation_entry_snap_t ^
 cn_rotation_exit_snap_t ^
 cn_rotation_holding_snap_t

REM ==========================================================
REM Recent daily tables with DATA_DATE uppercase
REM ==========================================================
set RECENT_DATA_DATE_TABLES=^
 cn_fund_etf_hist_em

REM ==========================================================
REM Recent event/fundamental tables by ann_date/end_date/asof_date
REM ==========================================================
set RECENT_ANN_DATE_TABLES=^
 cn_event_announcement_meta ^
 cn_event_dividend ^
 cn_event_earnings_express ^
 cn_event_earnings_forecast ^
 cn_event_fina_indicator ^
 cn_local_stock_balancesheet_q ^
 cn_local_stock_fina_indicator_q ^
 cn_local_stock_income_q

set RECENT_ASOF_DATE_TABLES=^
 cn_board_concept_cons ^
 cn_board_concept_member_stg ^
 cn_board_industry_cons ^
 cn_board_industry_member_stg

REM ==========================================================
REM START
REM ==========================================================

echo.
echo ==========================================================
echo RECENT MIRROR START: %SOURCE_DB% -^> %TARGET_DB%
echo RECENT_DAYS=%RECENT_DAYS%
echo ==========================================================

call :RUN_MYSQL "SELECT VERSION();"
if errorlevel 1 goto FAILED

REM Protect against accidental source drop
if /I "%SOURCE_DB%"=="%TARGET_DB%" (
  echo [ERROR] SOURCE_DB and TARGET_DB are identical. Abort.
  goto FAILED
)

echo.
echo [STEP 1] Recreate target schema...
call :RUN_MYSQL "DROP DATABASE IF EXISTS `%TARGET_DB%`; CREATE DATABASE `%TARGET_DB%` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;"
if errorlevel 1 goto FAILED

echo.
echo [STEP 2] Import schema, views, routines, triggers, events only...
mysqldump %DUMP_BASE% --no-data --routines --triggers --events %SOURCE_DB% | mysql %MYSQL_BASE% %TARGET_DB%
if errorlevel 1 (
  echo [ERROR] Schema import failed.
  goto FAILED
)

echo.
echo [STEP 3] Sync FULL small/dimension/config tables...
for %%T in (%FULL_TABLES%) do (
  call :SYNC_FULL %%T
  if errorlevel 1 goto FAILED
)

echo.
echo [STEP 4] Sync RECENT tables by trade_date...
for %%T in (%RECENT_TRADE_DATE_TABLES%) do (
  call :SYNC_RECENT %%T trade_date
  if errorlevel 1 goto FAILED
)

echo.
echo [STEP 5] Sync RECENT tables by TRADE_DATE...
for %%T in (%RECENT_TRADE_DATE_UPPER_TABLES%) do (
  call :SYNC_RECENT %%T TRADE_DATE
  if errorlevel 1 goto FAILED
)

echo.
echo [STEP 6] Sync RECENT tables by DATA_DATE...
for %%T in (%RECENT_DATA_DATE_TABLES%) do (
  call :SYNC_RECENT %%T DATA_DATE
  if errorlevel 1 goto FAILED
)

echo.
echo [STEP 7] Sync RECENT event/fundamental tables by ann_date...
for %%T in (%RECENT_ANN_DATE_TABLES%) do (
  call :SYNC_RECENT %%T ann_date
  if errorlevel 1 goto FAILED
)

echo.
echo [STEP 8] Sync RECENT membership staging/history tables by ASOF/asof date...
for %%T in (%RECENT_ASOF_DATE_TABLES%) do (
  call :SYNC_RECENT %%T ASOF_DATE
  if errorlevel 1 goto FAILED
)

echo.
echo [STEP 9] Verify target object count...
call :RUN_MYSQL "SELECT table_type, COUNT(*) FROM information_schema.tables WHERE table_schema='%TARGET_DB%' GROUP BY table_type;"
if errorlevel 1 goto FAILED

echo.
echo ==========================================================
echo RECENT MIRROR SUCCESS
echo ==========================================================
set MYSQL_PWD=
exit /b 0

REM ==========================================================
REM Functions
REM ==========================================================

:RUN_MYSQL
mysql %MYSQL_BASE% -e %~1
if errorlevel 1 exit /b 1
exit /b 0

:SYNC_FULL
set TBL=%~1
echo [FULL] !TBL!
call :TABLE_EXISTS !TBL!
if errorlevel 1 (
  echo [WARN] Source table !TBL! does not exist. Skip.
  exit /b 0
)
mysqldump %DUMP_BASE% --no-create-info --skip-triggers %SOURCE_DB% !TBL! | mysql %MYSQL_BASE% %TARGET_DB%
if errorlevel 1 (
  echo [ERROR] FULL sync failed: !TBL!
  exit /b 1
)
exit /b 0

:SYNC_RECENT
set TBL=%~1
set COL=%~2
echo [RECENT] !TBL! WHERE !COL! ^>= CURDATE() - INTERVAL %RECENT_DAYS% DAY
call :TABLE_EXISTS !TBL!
if errorlevel 1 (
  echo [WARN] Source table !TBL! does not exist. Skip.
  exit /b 0
)
call :COLUMN_EXISTS !TBL! !COL!
if errorlevel 1 (
  echo [ERROR] Column !COL! does not exist on !TBL!. Add override or move table to correct group.
  exit /b 1
)
mysqldump %DUMP_BASE% --no-create-info --skip-triggers %SOURCE_DB% !TBL! --where="!COL! >= DATE_SUB(CURDATE(), INTERVAL %RECENT_DAYS% DAY)" | mysql %MYSQL_BASE% %TARGET_DB%
if errorlevel 1 (
  echo [ERROR] RECENT sync failed: !TBL!
  exit /b 1
)
exit /b 0

:TABLE_EXISTS
set TBL=%~1
mysql %MYSQL_BASE% -N -B -e "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='%SOURCE_DB%' AND table_name='!TBL!' AND table_type='BASE TABLE';" > "%TEMP%\ga_table_exists.txt"
if errorlevel 1 exit /b 1
set /p EXISTS=<"%TEMP%\ga_table_exists.txt"
if not "!EXISTS!"=="1" exit /b 1
exit /b 0

:COLUMN_EXISTS
set TBL=%~1
set COL=%~2
mysql %MYSQL_BASE% -N -B -e "SELECT COUNT(*) FROM information_schema.columns WHERE table_schema='%SOURCE_DB%' AND table_name='!TBL!' AND column_name='!COL!';" > "%TEMP%\ga_column_exists.txt"
if errorlevel 1 exit /b 1
set /p EXISTS=<"%TEMP%\ga_column_exists.txt"
if not "!EXISTS!"=="1" exit /b 1
exit /b 0

:FAILED
echo.
echo ==========================================================
echo RECENT MIRROR FAILED
echo ==========================================================
set MYSQL_PWD=
exit /b 1
