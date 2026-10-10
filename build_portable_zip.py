"""Build a versioned, self-contained Windows ZIP without local research data."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from app import __version__

DOCS = ("实验模式.md", "量表乱序规则.md", "便携发布-1.5.10.md", "答案解析规则.md")
REQUIRED_FILES = ("main.py", "launcher.py", "launch.bat", "README.md", "平台使用手册.md",
                  "LICENSE", "VERSION", "requirements.txt", "requirements-lock.txt",
                  "LLM-Cultural-Tendencies.exe", "大语言模型潜在文化倾向性研究.exe", "runtime/python.exe",
                  "app/webapp.py", "app/static/app.ico")
EXCLUDED_DIRS = {"__pycache__", ".git", ".idea", "Scripts"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".bak", ".tmp"}
STORED_SUFFIXES = {".zip", ".exe", ".dll", ".pyd", ".cat", ".ico", ".jpg", ".jpeg", ".png", ".gif"}


def project_files(root: Path = ROOT) -> list[Path]:
    files = []
    for rel_root in (Path("app"), Path("runtime")):
        if not (root / rel_root).is_dir():
            raise ValueError(f"Missing directory: {rel_root}")
        for path in (root / rel_root).rglob("*"):
            rel = path.relative_to(root)
            if (path.is_file() and not any(part in EXCLUDED_DIRS for part in rel.parts)
                    and path.suffix.casefold() not in SKIP_SUFFIXES and path.name != "get-pip.py"):
                files.append(rel)
    files.extend(Path(name) for name in REQUIRED_FILES)
    files.extend(Path("docs") / name for name in DOCS)
    files.append(Path("scripts/reparse_results.py"))
    return sorted(set(files), key=lambda path: path.as_posix().casefold())


def validate_runtime(root: Path) -> None:
    pth_files = list((root / "runtime").glob("python*._pth"))
    if not pth_files:
        raise ValueError("Embedded Python path configuration is missing")
    for path in pth_files:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "import ")):
                continue
            if PureWindowsPath(line).drive or PureWindowsPath(line).root or Path(line).is_absolute():
                raise ValueError(f"Runtime contains an absolute import path: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-launcher-build", action="store_true", help="Use an already verified launcher")
    args = parser.parse_args()
    if (ROOT / "VERSION").read_text(encoding="utf-8").strip() != __version__:
        raise SystemExit("VERSION does not match app/__init__.py")
    validate_runtime(ROOT)
    if not args.skip_launcher_build:
        subprocess.run([str(ROOT / "runtime/python.exe"), str(ROOT / "build_gui_launcher.py")], cwd=ROOT, check=True)
    files = project_files()
    missing = [str(path) for path in files if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("Missing portable files: " + ", ".join(missing))
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    archive = dist / f"LLM-Cultural-Tendencies-Windows-x64-v{__version__}.zip"
    if archive.exists():
        raise SystemExit(f"Existing versioned ZIP must be preserved: {archive}")
    prefix = f"LLM-Cultural-Tendencies-v{__version__}/"
    manifest = {"version": __version__, "platform": "Windows x64", "files": {}}
    placeholders = {
        "data/README.txt": "请放入你自己的量表 Excel。公开包不包含研究量表。\nPlace your own questionnaire .xlsx files here.\n",
        "results/README.txt": "模型回答保存在此目录。\nModel responses are saved here.\n",
        "logs/README.txt": "运行日志保存在此目录。\nExecution logs are saved here.\n",
        "START_HERE.txt": f"LLM Cultural Tendencies Platform v{__version__}\n"
            "先解压完整压缩包，再双击 LLM-Cultural-Tendencies.exe 或同目录的中文启动器。\n"
            "无需安装 Python；目录可含中文、英文和空格。请勿只移动 EXE。\n"
            "量表、API 密钥、结果不附带，请填写自己的密钥并将量表放入 data。\n"
            "Extract the entire ZIP. Double-click LLM-Cultural-Tendencies.exe.\n"
            "Python is bundled. Keep all files together. Put your own scales in data.\n",
    }
    with zipfile.ZipFile(archive, "w", allowZip64=True) as bundle:
        for rel in files:
            source = ROOT / rel
            manifest["files"][rel.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
            method = zipfile.ZIP_STORED if source.suffix.casefold() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            bundle.write(source, prefix + rel.as_posix(), compress_type=method)
        for name, value in placeholders.items():
            bundle.writestr(prefix + name, value.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)
        bundle.writestr(prefix + "bundle-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert not bundle.testzip(), "Corrupt ZIP member"
        assert all(not name.casefold().endswith((".xlsx", "user_config.json")) for name in names)
        assert all(prefix + name in names for name in REQUIRED_FILES)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(f"Portable ZIP: {archive}\nBytes: {archive.stat().st_size}\nSHA256: {digest}")


if __name__ == "__main__":
    main()
