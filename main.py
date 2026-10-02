#!/usr/bin/env python3
"""大语言模型潜在文化倾向性研究 — 启动本地 Web 服务并自动打开浏览器。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.proxy_util import apply_proxy_to_env  # noqa: E402
from app.webapp import run_web  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="启动本地文化倾向性研究工作台")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="仅启动本地服务，不自动打开浏览器",
    )
    args = parser.parse_args()
    apply_proxy_to_env()
    run_web(host="127.0.0.1", port=8765, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
