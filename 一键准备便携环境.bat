@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在准备项目内便携 Python 环境（首次需要联网下载）...
echo 工作目录: %CD%
echo.

REM 优先用已有相对路径 runtime，否则用系统 python
if exist "runtime\python.exe" (
  "runtime\python.exe" "prepare_runtime.py"
) else (
  python "prepare_runtime.py"
)
if errorlevel 1 (
  echo 准备失败，请确认本机已临时安装 Python 且可访问 python.org
  pause
  exit /b 1
)
echo.
echo 完成。之后分发时请连同 runtime 文件夹一起拷贝。
echo 用户端双击「大语言模型潜在文化倾向性研究.exe」即可，无需再装 Python。
pause
