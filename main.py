#!/usr/bin/env python3
"""大语言模型潜在文化倾向性研究 — 启动本地 Web 服务并自动打开浏览器。"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.proxy_util import apply_proxy_to_env  # noqa: E402
from app.webapp import run_web  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Cultural Tendencies Platform")
    parser.add_argument("--no-browser", action="store_true", help="Start without opening a browser")
    parser.add_argument("--port", type=int, default=8765, help="Local port; 0 selects an available port")
    args = parser.parse_args()
    if not 0 <= args.port <= 65515:
        parser.error("port must be between 0 and 65515")
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    apply_proxy_to_env()
    run_web(host="127.0.0.1", port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
