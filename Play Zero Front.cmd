@echo off
setlocal
set "ELECTRON_RUN_AS_NODE="
if not exist "%~dp0prototype\node_modules\electron\dist\electron.exe" (
  echo Install the game dependencies first: cd prototype ^&^& npm install
  pause
  exit /b 1
)
start "Zero Front" "%~dp0prototype\node_modules\electron\dist\electron.exe" "%~dp0prototype"
