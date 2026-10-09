"""Extract and start the actual Windows ZIP with no system Python on PATH."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from openpyxl import Workbook

OPENER = build_opener(ProxyHandler({}))


def request_json(url: str, data: dict | None = None, timeout: float = 2) -> dict:
    request = Request(url, data=None if data is None else json.dumps(data).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    with OPENER.open(request, timeout=timeout) as response:
        return json.load(response)


def expected_workspace(root: Path) -> str:
    return hashlib.sha256(str(root.resolve()).casefold().encode("utf-8")).hexdigest()


def wait_for_platform(root: Path, port: int, version: str) -> tuple[str, dict]:
    deadline = time.monotonic() + 60
    last_identity = None
    while time.monotonic() < deadline:
        urls = []
        for name in ("verification-startup.log", "startup.log"):
            path = root / "logs" / name
            if path.exists():
                urls.extend(re.findall(r"http://127\.0\.0\.1:\d+", path.read_text(encoding="utf-8", errors="replace")))
        urls.extend(f"http://127.0.0.1:{candidate}" for candidate in range(port, port + 20))
        for url in dict.fromkeys(urls):
            try:
                value = request_json(url + "/api/health", timeout=0.2)
                last_identity = value
                if (value.get("app_id") == "llm-cultural-tendencies" and value.get("version") == version
                        and value.get("workspace_id") == expected_workspace(root)):
                    return url, value
            except (OSError, ValueError):
                pass
        time.sleep(0.1)
    raise RuntimeError(f"Extracted platform did not become reachable: {root}; requested port={port}; last identity={last_identity}")


class OtherService(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"app_id":"unrelated-service"}')

    def log_message(self, *args):
        pass


def check_copy(archive: Path, base: Path, label: str, version: str, use_launcher: bool) -> dict:
    print(f"Checking extracted ZIP: {label}", flush=True)
    destination = base / label
    destination.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    root = destination / f"LLM-Cultural-Tendencies-v{version}"
    if use_launcher:
        renamed = destination / "改名后的平台 中文 English"
        assert root.resolve().is_relative_to(base.resolve()) and renamed.resolve().is_relative_to(base.resolve())
        root.rename(renamed)
        root = renamed
    for rel in ("data", "results", "logs"):
        assert (root / rel).is_dir()
    assert not (root / "user_config.json").exists()
    manifest = json.loads((root / "bundle-manifest.json").read_text(encoding="utf-8"))
    for rel, digest in manifest["files"].items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == digest, rel
    workbook = Workbook()
    workbook.active.title = "Sheet1"
    chinese = workbook.create_sheet("Sheet2")
    for sheet in (workbook.active, chinese):
        sheet.append(["Synthetic portability fixture"])
        sheet.append(["Please answer the question."])
        sheet.append(["1. Synthetic question."])
    workbook.save(root / "data" / "Synthetic 中文 scale.xlsx")
    env = dict(os.environ)
    env["PATH"] = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32")
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    env["PYTHONUTF8"] = "1"
    runtime = root / "runtime/python.exe"
    imported = subprocess.run([str(runtime), "-c", "import flask, openai, openpyxl, requests; print('bundled imports OK')"],
                              cwd=base, env=env, capture_output=True, timeout=20)
    assert imported.returncode == 0, imported.stderr.decode("utf-8", errors="replace")
    blocker = ThreadingHTTPServer(("127.0.0.1", 0), OtherService) if use_launcher else None
    if blocker:
        threading.Thread(target=blocker.serve_forever, daemon=True).start()
        port = blocker.server_port
    else:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    command = ([str(root / "LLM-Cultural-Tendencies.exe")] if use_launcher
               else [str(runtime), str(root / "main.py")]) + ["--no-browser", "--port", str(port)]
    output_path = root / "logs/verification-startup.log"
    health = None
    with output_path.open("wb") as output:
        process = subprocess.Popen(command, cwd=base, env=env, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            url, health = wait_for_platform(root, port, version)
            for route in ("/", "/console", "/chat", "/static/css/style.css", "/static/js/console.js", "/static/bg.jpg"):
                with OPENER.open(url + route, timeout=3) as response:
                    assert response.status == 200, route
            loaded = request_json(url + "/api/scales")
            assert len(loaded["scales"]) == 1
            assert loaded["scales"][0]["en_n"] == loaded["scales"][0]["ch_n"] == 1
            assert request_json(url + "/api/config", {"data_dir": "./data", "results_dir": "./results"})["ok"]
            assert (root / "user_config.json").is_file()
            assert request_json(url + "/api/meta")["version"] == version
            if use_launcher:
                assert url != f"http://127.0.0.1:{port}", "Opened the unrelated occupied port"
                repeated = subprocess.run([str(root / "大语言模型潜在文化倾向性研究.exe"), "--no-browser", "--port", str(port)],
                                          cwd=base, env=env, timeout=10)
                assert repeated.returncode == 0
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    log = (root / "logs/startup.log").read_text(encoding="utf-8", errors="replace")
                    if "同一目录的平台已经运行" in log:
                        break
                    time.sleep(0.1)
                else:
                    raise AssertionError("Second launcher did not reuse the running platform")
                assert request_json(url + "/api/health")["pid"] == health["pid"]
            return {"path_case": label, "launcher": use_launcher, "renamed": use_launcher,
                    "system_python_on_path": False, "version": version, "status": "passed",
                    "occupied_port_handled": bool(blocker), "manifest_files": len(manifest["files"])}
        except Exception:
            output.flush()
            print(output_path.read_text(encoding="utf-8", errors="replace"), flush=True)
            gui_log = root / "logs/startup.log"
            if gui_log.exists():
                print(gui_log.read_text(encoding="utf-8", errors="replace"), flush=True)
            raise
        finally:
            if health and health.get("workspace_id") == expected_workspace(root):
                os.kill(health["pid"], signal.SIGTERM)
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
            if blocker:
                blocker.shutdown()
                blocker.server_close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    with zipfile.ZipFile(archive) as bundle:
        version = json.loads(bundle.read(next(name for name in bundle.namelist() if name.endswith("/bundle-manifest.json"))))["version"]
        assert not bundle.testzip()
        assert all(not name.casefold().endswith((".xlsx", "user_config.json")) for name in bundle.namelist())
    with tempfile.TemporaryDirectory(prefix="LLM portable English ") as folder:
        base = Path(folder)
        cases = [check_copy(archive, base, "English path with spaces", version, False),
                 check_copy(archive, base, "中文 解压路径", version, True)]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"version": version, "cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(cases, ensure_ascii=False))


if __name__ == "__main__":
    main()
