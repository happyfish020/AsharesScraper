@echo off
setlocal
if "%~2"=="" exit /b 2
set START=%1
set END=%2
shift /1
shift /1
for /L %%Y in (%START%,1,%END%) do (
  call "%~dp0daily.bat" %%Y %*
  if errorlevel 1 exit /b 1
)
exit /b 0
