@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   环境检查（全部使用相对路径）
echo   当前目录: %CD%
echo ============================================================
echo.

if exist "runtime\python.exe" (
  echo [OK] runtime\python.exe
) else (
  echo [缺] runtime\python.exe  ^<-- 这是双击启动失败的常见原因
)

if exist "main.py" (
  echo [OK] main.py
) else (
  echo [缺] main.py
)

if exist "app\webapp.py" (
  echo [OK] app\webapp.py
) else (
  echo [缺] app\webapp.py
)

if exist "data" (
  echo [OK] data\
) else (
  echo [缺] data\
)

if exist "runtime\Lib\site-packages\flask" (
  echo [OK] runtime\Lib\site-packages\flask
) else (
  echo [缺] flask 依赖  ^<-- 请运行 一键准备便携环境.bat
)

echo.
echo 若上面全部 [OK]，请双击「大语言模型潜在文化倾向性研究.exe」启动。
echo.
pause
