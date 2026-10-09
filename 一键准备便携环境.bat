@echo off
cd /d "%~dp0"
echo ============================================================
echo   Preparing the bundled portable Python environment
echo   (First run needs an internet connection to download)
echo   Working dir: %CD%
echo ============================================================
echo.

REM Prefer the bundled relative-path runtime; otherwise use system python
if exist "runtime\python.exe" (
  "runtime\python.exe" "prepare_runtime.py"
) else (
  python "prepare_runtime.py"
)
if errorlevel 1 (
  echo [ERROR] Prepare failed. Make sure Python is temporarily installed
  echo         and python.org is reachable, then try again.
  pause
  exit /b 1
)
echo.
echo [DONE] When distributing, copy the whole folder including runtime.
echo  End users just double-click the EXE - no need to install Python.
pause