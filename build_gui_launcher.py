# -*- coding: utf-8 -*-
"""Build a portable GUI launcher EXE with custom icon (relative paths)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ICON = ROOT / "app" / "static" / "app.ico"
OUT_NAME = "大语言模型潜在文化倾向性研究"
LAUNCHER_PY = ROOT / "_gui_launcher.py"


def main() -> None:
    if not ICON.exists():
        raise SystemExit(f"Missing icon: {ICON}")

    # Starts a new console for the server window; this EXE itself is windowed (icon on desktop).
    LAUNCHER_PY.write_text(
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "CREATE_NEW_CONSOLE = 0x00000010\n"
        "\n"
        "def main() -> None:\n"
        "    if getattr(sys, 'frozen', False):\n"
        "        root = Path(sys.executable).resolve().parent\n"
        "    else:\n"
        "        root = Path(__file__).resolve().parent\n"
        "    os.chdir(root)\n"
        "    bat = root / 'launch.bat'\n"
        "    py = root / 'runtime' / 'python.exe'\n"
        "    main_py = root / 'main.py'\n"
        "    if bat.exists():\n"
        "        subprocess.Popen(\n"
        "            ['cmd.exe', '/c', str(bat)],\n"
        "            cwd=str(root),\n"
        "            creationflags=CREATE_NEW_CONSOLE,\n"
        "        )\n"
        "        return\n"
        "    if py.exists() and main_py.exists():\n"
        "        subprocess.Popen(\n"
        "            [str(py), str(main_py)],\n"
        "            cwd=str(root),\n"
        "            creationflags=CREATE_NEW_CONSOLE,\n"
        "        )\n"
        "        return\n"
        "    import ctypes\n"
        "    ctypes.windll.user32.MessageBoxW(\n"
        "        0,\n"
        "        'Missing launch.bat or runtime\\\\python.exe',\n"
        "        'Launch failed',\n"
        "        0x10,\n"
        "    )\n"
        "    raise SystemExit(1)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )

    py = ROOT / "runtime" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)

    cmd = [
        str(py),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        f"--name={OUT_NAME}",
        f"--icon={ICON}",
        "--distpath",
        str(ROOT),
        "--workpath",
        str(ROOT / "build_launcher"),
        "--specpath",
        str(ROOT / "build_launcher"),
        str(LAUNCHER_PY),
    ]
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)
    LAUNCHER_PY.unlink(missing_ok=True)
    print("Built:", ROOT / f"{OUT_NAME}.exe")


if __name__ == "__main__":
    main()
