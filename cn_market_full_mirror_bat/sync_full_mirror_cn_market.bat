@echo off
setlocal EnableExtensions

REM ==========================================================
REM GrowthAlpha cn_market Full Mirror Sync
REM Source : cn_market_red
REM Target : cn_market
REM Mode   : full schema + full data, streamed; no dump file written.
REM ==========================================================

set MYSQL_HOST=127.0.0.1
set MYSQL_PORT=3306
set MYSQL_USER=root
set MYSQL_PWD=YOUR_PASSWORD_HERE

set SOURCE_DB=cn_market_red
set TARGET_DB=cn_market

set MYSQL_BASE=-h %MYSQL_HOST% -P %MYSQL_PORT% -u %MYSQL_USER% --default-character-set=utf8mb4
set DUMP_BASE=-h %MYSQL_HOST% -P %MYSQL_PORT% -u %MYSQL_USER% --default-character-set=utf8mb4 --single-transaction --routines --triggers --events --set-gtid-purged=OFF --no-tablespaces

echo.
echo ==========================================================
echo FULL MIRROR START: %SOURCE_DB% -^> %TARGET_DB%
echo ==========================================================

mysql %MYSQL_BASE% -e "SELECT VERSION();"
if errorlevel 1 goto FAILED

if /I "%SOURCE_DB%"=="%TARGET_DB%" (
  echo [ERROR] SOURCE_DB and TARGET_DB are identical. Abort.
  goto FAILED
)

echo.
echo [STEP 1] Recreate target schema...
mysql %MYSQL_BASE% -e "DROP DATABASE IF EXISTS `%TARGET_DB%`; CREATE DATABASE `%TARGET_DB%` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;"
if errorlevel 1 goto FAILED

echo.
echo [STEP 2] Stream full dump/import...
mysqldump %DUMP_BASE% %SOURCE_DB% | mysql %MYSQL_BASE% %TARGET_DB%
if errorlevel 1 goto FAILED

echo.
echo [STEP 3] Verify target object count...
mysql %MYSQL_BASE% -e "SELECT table_type, COUNT(*) FROM information_schema.tables WHERE table_schema='%TARGET_DB%' GROUP BY table_type;"
if errorlevel 1 goto FAILED

echo.
echo ==========================================================
echo FULL MIRROR SUCCESS
echo ==========================================================
set MYSQL_PWD=
exit /b 0

:FAILED
echo.
echo ==========================================================
echo FULL MIRROR FAILED
echo ==========================================================
set MYSQL_PWD=
exit /b 1
