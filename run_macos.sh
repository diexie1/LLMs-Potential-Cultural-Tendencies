#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ -x "./.venv/bin/python" ]; then
  exec ./.venv/bin/python main.py
elif [ -x "./runtime/bin/python" ]; then
  ./runtime/bin/python main.py
elif [ -x "./runtime/bin/python3" ]; then
  ./runtime/bin/python3 main.py
else
  echo "未找到项目环境。请先执行: sh setup.sh"
  exit 1
fi
