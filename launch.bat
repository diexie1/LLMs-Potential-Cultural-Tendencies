@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist "main.py" (
  echo [ERROR] Missing: main.py
  pause
  exit /b 1
)

if exist ".venv\Scripts\python.exe" (
  echo Starting with project environment...
  ".venv\Scripts\python.exe" "main.py"
  goto :finished
)

if exist "runtime\python.exe" (
  echo Starting with bundled runtime...
  "runtime\python.exe" "main.py"
  goto :finished
)

where py >nul 2>nul
if not errorlevel 1 (
  echo Starting with Python Launcher...
  py -3 "main.py"
  goto :finished
)

where python >nul 2>nul
if not errorlevel 1 (
  echo Starting with Python...
  python "main.py"
  goto :finished
)

echo [ERROR] Python was not found.
echo Run setup_windows.bat first, or download the Windows portable package from GitHub Releases.
pause
exit /b 1

:finished
if errorlevel 1 pause
endlocal
