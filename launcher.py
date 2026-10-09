"""Portable Windows launcher; resolve every file beside this executable."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    python = root / "runtime" / "python.exe"
    entry = root / "main.py"
    if not python.is_file() or not entry.is_file():
        ctypes.windll.user32.MessageBoxW(
            0, "请先解压完整的平台文件夹，再从文件夹中启动。\n"
            "Please extract the entire ZIP before starting.\nMissing runtime/python.exe or main.py.",
            "Platform startup", 0x10)
        raise SystemExit(1)
    env = dict(os.environ, PYTHONUTF8="1")
    command = [str(python), str(entry), *sys.argv[1:]]
    if "--no-browser" in sys.argv:
        (root / "logs").mkdir(exist_ok=True)
        with (root / "logs/startup.log").open("ab") as output:
            subprocess.Popen(command, cwd=str(root), env=env, creationflags=subprocess.CREATE_NO_WINDOW,
                             stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
    else:
        subprocess.Popen(command, cwd=str(root), env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)


if __name__ == "__main__":
    main()
