@echo off
setlocal EnableExtensions DisableDelayedExpansion
if "%~2"=="" (
    echo Usage: daily_by_year.bat START_YEAR END_YEAR [--refresh]
    exit /b 2
)
set "START_YEAR=%~1"
set "END_YEAR=%~2"
shift /1
shift /1
for /L %%Y in (%START_YEAR%,1,%END_YEAR%) do (
    call "%~dp0daily.bat" %%Y %*
    if errorlevel 1 exit /b 1
)
exit /b 0
