@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PY_CMD="
where py >nul 2>nul
if not errorlevel 1 set "PY_CMD=py -3"
if not defined PY_CMD (
  where python >nul 2>nul
  if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD (
  echo [ERROR] Python 3.10 or newer is required.
  echo Install Python from https://www.python.org/downloads/ and run this file again.
  goto :failed
)

%PY_CMD% -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python 3.10 or newer was not found. Install a current Python version and try again.
  goto :failed
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the project environment...
  %PY_CMD% -m venv .venv
  if errorlevel 1 goto :failed
)

echo Installing platform dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
if errorlevel 1 goto :failed

echo.
echo Setup complete. Double-click launch.bat to start the platform.
pause
endlocal
exit /b 0

:failed
echo.
echo Setup failed. Check your Python installation and internet connection, then retry.
pause
endlocal
exit /b 1
