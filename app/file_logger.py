"""线程安全的运行日志写入。"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


class RunFileLogger:
    """将每次调用的请求/响应等详细信息写入 log 文件。"""

    def __init__(self, log_path: Path):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from .config import display_path

            self._display = display_path(self.path)
        except Exception:
            self._display = str(self.path)
        self._lock = threading.Lock()
        self._write_raw(
            f"{'=' * 72}\n"
            f"日志开始: {datetime.now().isoformat(timespec='seconds')}\n"
            f"文件: {self._display}\n"
            f"{'=' * 72}\n"
        )

    def _write_raw(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self._write_raw(f"[{ts}] {message}")

    def log_call(
        self,
        *,
        scale_name: str,
        language: str,
        order_id: int,
        attempt: int,
        prompt: str,
        response: Optional[str] = None,
        scores: Optional[list] = None,
        result_path: Optional[str] = None,
        error: Optional[str] = None,
        seed: Optional[int] = None,
        shuffle_order: Optional[list] = None,
    ) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "",
            "-" * 72,
            f"[{ts}] CALL {scale_name}/{language} order_id={order_id} attempt={attempt}",
        ]
        if seed is not None:
            lines.append(f"shuffle_seed={seed}")
        if shuffle_order is not None:
            lines.append(f"shuffle_order={shuffle_order}")
        lines.append("[PROMPT]")
        lines.append(prompt)
        if response is not None:
            lines.append("[RESPONSE]")
            lines.append(response)
        if scores is not None:
            lines.append(f"[SCORES_ORIGINAL_ORDER] {scores}")
        if result_path:
            lines.append(f"[SAVED] {result_path}")
        if error:
            lines.append(f"[ERROR] {error}")
        lines.append("-" * 72)
        with self._lock:
            self._write_raw("\n".join(lines) + "\n")

    def close_note(self, message: str = "日志结束") -> None:
        self.log(message)
