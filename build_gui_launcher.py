"""Build the relative-path launcher, with matching Windows version metadata."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from app import __version__

OUT_NAME = "LLM-Cultural-Tendencies"


def main() -> None:
    if (ROOT / "VERSION").read_text(encoding="utf-8").strip() != __version__:
        raise SystemExit("VERSION does not match app/__init__.py")
    work = ROOT / "build_launcher"
    work.mkdir(exist_ok=True)
    version_file = work / "windows-version.txt"
    numbers = tuple(int(part) for part in __version__.split(".")) + (0,)
    version_file.write_text(f'''VSVersionInfo(
  ffi=FixedFileInfo(filevers={numbers!r}, prodvers={numbers!r}, mask=0x3f,
    flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('040904B0', [
    StringStruct('FileDescription', 'LLM Cultural Tendencies Platform'),
    StringStruct('FileVersion', '{__version__}.0'),
    StringStruct('ProductVersion', '{__version__}'),
    StringStruct('ProductName', 'LLM Cultural Tendencies Platform'),
    StringStruct('OriginalFilename', '{OUT_NAME}.exe')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])])
''', encoding="utf-8")
    python = ROOT / "runtime/python.exe"
    subprocess.run([str(python if python.is_file() else Path(sys.executable)), "-m", "PyInstaller",
                    "--noconfirm", "--clean", "--onefile", "--windowed", f"--name={OUT_NAME}",
                    f"--icon={ROOT / 'app/static/app.ico'}", f"--version-file={version_file}",
                    "--distpath", str(ROOT), "--workpath", str(work), "--specpath", str(work),
                    str(ROOT / "launcher.py")], cwd=ROOT, check=True)
    output = ROOT / f"{OUT_NAME}.exe"
    shutil.copy2(output, ROOT / "大语言模型潜在文化倾向性研究.exe")
    print(f"Built launcher {__version__}: {output}")


if __name__ == "__main__":
    main()
