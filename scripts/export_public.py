"""Export reviewed tracked platform files into an existing public Git clone."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOTS = {"app", "scripts", "tests", ".github"}
PUBLIC_FILES = {".gitignore", ".gitattributes", "LICENSE", "VERSION", "README.md", "平台使用手册.md",
                "main.py", "launcher.py", "launch.bat", "setup_windows.bat", "setup.sh", "run_macos.sh",
                "prepare_runtime.py", "build_gui_launcher.py", "build_portable_zip.py",
                "requirements.txt", "requirements-lock.txt", "一键准备便携环境.bat", "检查环境.bat"}
PUBLIC_DOCS = {"docs/实验模式.md", "docs/量表乱序规则.md", "docs/便携发布-1.5.6.md", "docs/便携发布-1.5.10.md", "docs/答案解析规则.md"}
SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|AIza[A-Za-z0-9_-]{30,})\b")
OBSOLETE_FILES = {"scale_app.spec"}


def tracked(folder: Path) -> set[str]:
    output = subprocess.check_output(["git", "-C", str(folder), "ls-files", "-z"])
    return {name for name in output.decode("utf-8").split("\0") if name}


def is_public(name: str) -> bool:
    path = Path(name)
    return name in PUBLIC_FILES | PUBLIC_DOCS | OBSOLETE_FILES or path.parts[0] in PUBLIC_ROOTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    target = args.target.resolve()
    if target == ROOT or not (target / ".git").is_dir():
        raise SystemExit("Target must be a separate, existing Git clone")
    names = {name for name in tracked(ROOT) if is_public(name)}
    if not PUBLIC_FILES.issubset(names) or not PUBLIC_DOCS.issubset(names):
        raise SystemExit("Public source files must be tracked before export")
    for name in names:
        source = ROOT / name
        if source.suffix.casefold() not in {".jpg", ".png", ".gif", ".ico"}:
            if SECRET.search(source.read_text(encoding="utf-8-sig")):
                raise SystemExit(f"Potential credential found; export stopped: {name}")
    for name in tracked(target) - names:
        if is_public(name):
            destination = (target / name).resolve()
            if not destination.is_relative_to(target):
                raise SystemExit("Delete path escapes target checkout")
            destination.unlink(missing_ok=True)
    for name in names:
        destination = (target / name).resolve()
        if not destination.is_relative_to(target):
            raise SystemExit("Write path escapes target checkout")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    print(f"Exported {len(names)} reviewed platform files. Research data and private history excluded.")


if __name__ == "__main__":
    main()
