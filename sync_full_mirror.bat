REM sync_full_mirror.bat
@echo off
setlocal

set MYSQL_HOST=127.0.0.1
set MYSQL_PORT=3306
set MYSQL_USER=root
set MYSQL_PWD=YOUR_PASSWORD

set SOURCE_DB=cn_market_red
set TARGET_DB=cn_market

echo.
echo =====================================
echo FULL MIRROR START
echo =====================================

mysql ^
-h %MYSQL_HOST% ^
-P %MYSQL_PORT% ^
-u %MYSQL_USER% ^
-e "DROP DATABASE IF EXISTS %TARGET_DB%; CREATE DATABASE %TARGET_DB% DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;"

if errorlevel 1 goto FAILED

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
%SOURCE_DB% ^
| mysql ^
-h %MYSQL_HOST% ^
-P %MYSQL_PORT% ^
-u %MYSQL_USER% ^
--default-character-set=utf8mb4 ^
%TARGET_DB%

if errorlevel 1 goto FAILED

echo.
echo FULL MIRROR SUCCESS
goto END

:FAILED
echo.
echo FULL MIRROR FAILED
exit /b 1

:END
set MYSQL_PWD=
exit /b 0