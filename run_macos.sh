#!/bin/bash
cd "$(dirname "$0")"
if [ -x "./runtime/bin/python" ]; then
  ./runtime/bin/python main.py
elif [ -x "./runtime/bin/python3" ]; then
  ./runtime/bin/python3 main.py
elif [ -x "./.venv/bin/python" ]; then
  ./.venv/bin/python main.py
else
  echo "请先执行 sh setup.sh，再运行 sh run_macos.sh。"
  exit 1
fi
