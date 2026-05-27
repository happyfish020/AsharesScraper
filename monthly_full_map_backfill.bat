@echo off
setlocal
set "V8_MONTHLY_FULL_MAP_HIST=1"
call "%~dp0monthly.bat" %*
exit /b %ERRORLEVEL%
