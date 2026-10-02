@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在准备项目内便携 Python 环境（首次需要联网下载）...
echo 工作目录: %CD%
echo.

REM 优先使用已安装的项目运行时；否则使用 Python Launcher 或系统 Python。
if exist "runtime\python.exe" (
  "runtime\python.exe" "prepare_runtime.py"
) else (
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 "prepare_runtime.py"
  ) else (
    python "prepare_runtime.py"
  )
)
if errorlevel 1 (
  echo 准备失败，请确认本机有 Python 3.10+ 且可访问 python.org 和 PyPI。
  pause
  exit /b 1
)

echo 正在安装启动器构建依赖...
"runtime\python.exe" -m pip install pyinstaller
if errorlevel 1 (
  echo PyInstaller 安装失败。
  pause
  exit /b 1
)

echo 正在生成 Windows 启动程序...
"runtime\python.exe" "build_gui_launcher.py"
if errorlevel 1 (
  echo 启动程序构建失败。
  pause
  exit /b 1
)

echo.
echo 完成。分发时请连同 runtime 文件夹、启动程序和 data 目录一起拷贝。
echo 不要把 API 密钥、个人数据或无权公开的量表放入分发包。
pause
