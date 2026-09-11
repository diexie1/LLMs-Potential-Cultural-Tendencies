@echo off
REM Internal relative-path launcher. Prefer the desktop icon EXE:
REM   大语言模型潜在文化倾向性研究.exe
chcp 65001 >nul
cd /d "%~dp0"
if not exist "runtime\python.exe" (
  echo [ERROR] Missing: runtime\python.exe
  echo Current dir: %CD%
  pause
  exit /b 1
)
if not exist "main.py" (
  echo [ERROR] Missing: main.py
  pause
  exit /b 1
)
echo Starting with relative paths: runtime\python.exe main.py
"runtime\python.exe" "main.py"
if errorlevel 1 pause
