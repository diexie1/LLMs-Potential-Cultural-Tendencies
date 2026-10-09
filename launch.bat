@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "TASK_ROOT=%~dp0"
cd /d "%TASK_ROOT%"
if exist "%TASK_ROOT%runtime\python.exe" (
  "%TASK_ROOT%runtime\python.exe" "%TASK_ROOT%main.py" %*
) else if exist "%TASK_ROOT%.venv\Scripts\python.exe" (
  "%TASK_ROOT%.venv\Scripts\python.exe" "%TASK_ROOT%main.py" %*
) else (
  echo [ERROR] Runtime missing. Download the Windows portable ZIP from GitHub Releases.
  echo For source code: install Python, then run setup_windows.bat.
  pause
  exit /b 1
)
if errorlevel 1 pause
