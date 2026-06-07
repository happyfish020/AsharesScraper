@echo off
setlocal EnableExtensions

REM ==========================================================
REM MySQL Full Mirror Sync - cn_market_red -> cn_market
REM ----------------------------------------------------------
REM Purpose:
REM   Full weekly mirror without creating a dump file on disk.
REM   Uses streaming pipe: mysqldump | mysql
REM
REM Safety:
REM   - Stops immediately if any step fails
REM   - Source DB is never written
REM   - Target DB is dropped/recreated
REM
REM Requirements:
REM   - mysql.exe and mysqldump.exe are in PATH
REM   - Use an admin account such as root
REM ==========================================================

REM ====== USER CONFIG ======
set MYSQL_HOST=127.0.0.1
set MYSQL_PORT=3306
set MYSQL_USER=root
set MYSQL_PWD=YOUR_PASSWORD_HERE

set SOURCE_DB=cn_market_red
set TARGET_DB=cn_market

REM Recommended for MySQL 8/9 utf8mb4 projects
set TARGET_CHARSET=utf8mb4
set TARGET_COLLATION=utf8mb4_0900_ai_ci

REM ====== INTERNAL ======
set START_TS=%date% %time%

echo.
echo ==========================================================
echo MySQL Full Mirror Sync
echo Source: %SOURCE_DB%
echo Target: %TARGET_DB%
echo Started: %START_TS%
echo ==========================================================
echo.

if "%SOURCE_DB%"=="" (
    echo [ERROR] SOURCE_DB is empty.
    goto :FAILED
)

if "%TARGET_DB%"=="" (
    echo [ERROR] TARGET_DB is empty.
    goto :FAILED
)

if /I "%SOURCE_DB%"=="%TARGET_DB%" (
    echo [ERROR] SOURCE_DB and TARGET_DB must be different.
    goto :FAILED
)

where mysql >nul 2>nul
if errorlevel 1 (
    echo [ERROR] mysql.exe not found in PATH.
    goto :FAILED
)

where mysqldump >nul 2>nul
if errorlevel 1 (
    echo [ERROR] mysqldump.exe not found in PATH.
    goto :FAILED
)

echo ==========================================================
echo STEP 1 - Verify MySQL connection
echo ==========================================================
mysql ^
  -h %MYSQL_HOST% ^
  -P %MYSQL_PORT% ^
  -u %MYSQL_USER% ^
  --default-character-set=utf8mb4 ^
  -e "SELECT VERSION() AS mysql_version;"

if errorlevel 1 (
    echo [ERROR] Cannot connect to MySQL.
    goto :FAILED
)

echo.
echo ==========================================================
echo STEP 2 - Verify source database exists
echo ==========================================================
mysql ^
  -h %MYSQL_HOST% ^
  -P %MYSQL_PORT% ^
  -u %MYSQL_USER% ^
  --default-character-set=utf8mb4 ^
  -N -B ^
  -e "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME='%SOURCE_DB%';" | findstr /R /C:"^%SOURCE_DB%$" >nul

if errorlevel 1 (
    echo [ERROR] Source database %SOURCE_DB% does not exist.
    goto :FAILED
)

echo Source database exists: %SOURCE_DB%

echo.
echo ==========================================================
echo STEP 3 - Drop and recreate target database
echo WARNING: This will DELETE all objects in %TARGET_DB%.
echo ==========================================================
mysql ^
  -h %MYSQL_HOST% ^
  -P %MYSQL_PORT% ^
  -u %MYSQL_USER% ^
  --default-character-set=utf8mb4 ^
  -e "DROP DATABASE IF EXISTS `%TARGET_DB%`; CREATE DATABASE `%TARGET_DB%` DEFAULT CHARACTER SET %TARGET_CHARSET% COLLATE %TARGET_COLLATION%;"

if errorlevel 1 (
    echo [ERROR] Failed to drop/recreate target database %TARGET_DB%.
    goto :FAILED
)

echo Target database recreated: %TARGET_DB%

echo.
echo ==========================================================
echo STEP 4 - Stream full mirror: mysqldump ^| mysql
echo ==========================================================
mysqldump ^
  -h %MYSQL_HOST% ^
  -P %MYSQL_PORT% ^
  -u %MYSQL_USER% ^
  --default-character-set=utf8mb4 ^
  --single-transaction ^
  --routines ^
  --triggers ^
  --events ^
  --set-gtid-purged=OFF ^
  --no-tablespaces ^
  --skip-comments ^
  %SOURCE_DB% ^
| mysql ^
  -h %MYSQL_HOST% ^
  -P %MYSQL_PORT% ^
  -u %MYSQL_USER% ^
  --default-character-set=utf8mb4 ^
  %TARGET_DB%

if errorlevel 1 (
    echo [ERROR] Full mirror stream failed.
    goto :FAILED
)

echo.
echo ==========================================================
echo STEP 5 - Verify object count
echo ==========================================================
mysql ^
  -h %MYSQL_HOST% ^
  -P %MYSQL_PORT% ^
  -u %MYSQL_USER% ^
  --default-character-set=utf8mb4 ^
  -e "SELECT TABLE_SCHEMA, COUNT(*) AS object_count FROM information_schema.TABLES WHERE TABLE_SCHEMA IN ('%SOURCE_DB%', '%TARGET_DB%') GROUP BY TABLE_SCHEMA;"

if errorlevel 1 (
    echo [ERROR] Verification query failed.
    goto :FAILED
)

echo.
echo ==========================================================
echo SUCCESS - Full mirror completed
echo Started : %START_TS%
echo Finished: %date% %time%
echo ==========================================================

set MYSQL_PWD=
exit /b 0

:FAILED
echo.
echo ==========================================================
echo FAILED - Full mirror did not complete
echo Started : %START_TS%
echo Failed  : %date% %time%
echo ==========================================================
set MYSQL_PWD=
exit /b 1
