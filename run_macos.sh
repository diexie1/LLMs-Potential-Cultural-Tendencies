#!/bin/bash
cd "$(dirname "$0")"
if [ -x "./runtime/bin/python" ]; then
  ./runtime/bin/python main.py
elif [ -x "./runtime/bin/python3" ]; then
  ./runtime/bin/python3 main.py
else
  echo "未找到 runtime。请先执行: python3 prepare_runtime.py"
  exit 1
fi
