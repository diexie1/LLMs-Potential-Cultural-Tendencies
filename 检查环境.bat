@echo off
cd /d "%~dp0"
echo ============================================================
echo   Environment check (all relative paths)
echo   Current dir: %CD%
echo ============================================================
echo.

if exist "runtime\python.exe" (
  echo [OK] runtime\python.exe
) else (
  echo [MISSING] runtime\python.exe  ^<-- most common reason the EXE will not start
)

if exist "main.py" (
  echo [OK] main.py
) else (
  echo [MISSING] main.py
)

if exist "app\webapp.py" (
  echo [OK] app\webapp.py
) else (
  echo [MISSING] app\webapp.py
)

if exist "data" (
  echo [OK] data\
) else (
  echo [MISSING] data\
)

if exist "runtime\Lib\site-packages\flask" (
  echo [OK] runtime\Lib\site-packages\flask
) else (
  echo [MISSING] flask dependency  ^<-- run the Prepare-Environment setup script
)

echo.
echo If everything above is [OK], double-click the EXE to start.
echo.
pause