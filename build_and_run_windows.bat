@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Preserve configure/build/runtime diagnostics when this
rem launcher is double-clicked. PowerShell keeps a persistent transcript and
rem this wrapper pauses on every failure.
where powershell.exe >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Windows PowerShell was not found.
  echo Run this project from a Developer Command Prompt instead.
  pause
  exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\build_and_run_windows.ps1" %*
set "V47_RC=%ERRORLEVEL%"
if not "%V47_RC%"=="0" (
  echo.
  echo [Hair Studio] The build or application failed.
  echo Persistent diagnostics: "%~dp0logs\last_run.log"
  echo Runtime fatal record:  "%~dp0logs\last_fatal.txt" ^(if created^)
  echo.
  pause
)
exit /b %V47_RC%
