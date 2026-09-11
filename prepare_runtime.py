# -*- coding: utf-8 -*-
"""在项目内创建便携 Python 运行时（Windows），拷贝整个项目即可运行。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
# 使用官方 Windows embeddable 包，便于整夹迁移
PYTHON_VERSION = "3.12.8"
EMBED_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/"
    f"python-{PYTHON_VERSION}-embed-amd64.zip"
)
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


def _run(cmd: list[str], **kwargs) -> None:
    print(">", " ".join(cmd))
    subprocess.check_call(cmd, **kwargs)


def prepare() -> None:
    if sys.platform != "win32":
        print("当前脚本用于 Windows 便携环境。macOS 请在本机执行:")
        print("  python3 -m venv runtime && runtime/bin/pip install -r requirements.txt")
        print("  然后使用 run_macos.sh 启动。")
        # 仍尝试在非 Windows 上创建 venv 作为降级方案
        if not (RUNTIME / "bin" / "python").exists() and not (
            RUNTIME / "Scripts" / "python.exe"
        ).exists():
            _run([sys.executable, "-m", "venv", str(RUNTIME)])
            pip = RUNTIME / "bin" / "pip"
            _run([str(pip), "install", "--upgrade", "pip"])
            _run([str(pip), "install", "-r", str(ROOT / "requirements.txt")])
        print("完成:", RUNTIME)
        return

    RUNTIME.mkdir(parents=True, exist_ok=True)
    zip_path = RUNTIME / f"python-{PYTHON_VERSION}-embed-amd64.zip"
    python_exe = RUNTIME / "python.exe"

    if not python_exe.exists():
        print("下载便携 Python…")
        urllib.request.urlretrieve(EMBED_URL, zip_path)
        print("解压…")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RUNTIME)
        zip_path.unlink(missing_ok=True)

    # 启用 site，并加入相对 site-packages（不要写盘符绝对路径）
    pth_files = list(RUNTIME.glob("python*._pth"))
    if not pth_files:
        raise RuntimeError("未找到 python*._pth，嵌入式 Python 不完整")
    pth = pth_files[0]
    # 全部为相对 runtime 目录的路径
    pth.write_text(
        "python312.zip\n"
        ".\n"
        "Lib\\site-packages\n"
        "import site\n",
        encoding="utf-8",
    )

    get_pip = RUNTIME / "get-pip.py"
    if not (RUNTIME / "Scripts" / "pip.exe").exists() and not (
        RUNTIME / "Lib" / "site-packages" / "pip"
    ).exists():
        print("安装 pip…")
        urllib.request.urlretrieve(GET_PIP_URL, get_pip)
        _run([str(python_exe), str(get_pip), "--no-warn-script-location"])

    print("安装项目依赖到 runtime …")
    # embed 环境用 python -m pip
    _run(
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "--no-warn-script-location",
        ]
    )
    _run(
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "-r",
            str(ROOT / "requirements.txt"),
            "--no-warn-script-location",
        ]
    )

    # 清理缓存减小体积
    for p in RUNTIME.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    shutil.rmtree(RUNTIME / "Lib" / "site-packages" / "pip" / "__pycache__", ignore_errors=True)

    # Scripts\*.exe 会内嵌安装时的绝对路径；本项目从不依赖它们启动。
    # 写入相对路径包装脚本，需要时用: runtime\pip.bat / runtime\python.exe -m ...
    scripts = RUNTIME / "Scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "README_相对路径.txt").write_text(
        "本目录下的 *.exe 可能含有安装时的绝对路径，整夹迁移后可能失效。\n"
        "请始终双击项目根目录的「大语言模型潜在文化倾向性研究.exe」。\n"
        "内部通过相对路径调用: runtime\\python.exe main.py\n"
        "安装/升级包请用: ..\\python.exe -m pip install ...\n",
        encoding="utf-8",
    )
    (RUNTIME / "pip.bat").write_text(
        '@echo off\r\n'
        'cd /d "%~dp0"\r\n'
        '"%~dp0python.exe" -m pip %*\r\n',
        encoding="ascii",
    )

    print()
    print("=" * 60)
    print("便携环境已就绪:", RUNTIME)
    print("用户只需拷贝整个项目文件夹，双击「大语言模型潜在文化倾向性研究.exe」即可运行。")
    print("=" * 60)


if __name__ == "__main__":
    prepare()
