"""路径与默认配置：相对工程根目录，打包后仍可迁移。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path


def app_root() -> Path:
    """程序根目录：开发时为项目目录，打包后为可执行文件所在目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


ROOT = app_root()
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
CONFIG_PATH = ROOT / "user_config.json"
_CONFIG_LOCK = threading.RLock()

# 界面默认相对路径（相对程序根目录）
DEFAULT_DATA_DIR = "./data"
DEFAULT_RESULTS_DIR = "./results"

DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONDITION_MODE = "natural"
DEFAULT_CALL_COUNT = 100
DEFAULT_CONCURRENCY = 5
DEFAULT_SCALE_CONCURRENCY = 2
MAX_CONCURRENCY = 500
MAX_SCALE_CONCURRENCY = 500
DEFAULT_MAX_RETRIES = 10
DEFAULT_RETRY_DELAY_SEC = 2.0
DEFAULT_CAPTURE_NETWORK = True
DEFAULT_CAPTURE_MODEL_CATALOG = True
DEFAULT_NETWORK_GUARD = "warn"

# 默认端点；API 密钥只从被忽略的 user_config.json 或界面配置读取。
DEFAULT_API_KEYS = {
    "deepseek": {
        "api_key": "",
        "base_url": "https://api.deepseek.com",
    },
    "openai": {
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
    },
    "qwen": {
        "api_key": "",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "gemini": {
        "api_key": "",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}

DEFAULT_MODELS = {
    "deepseek": ["deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o3-mini"],
    "qwen": ["qwen-plus", "qwen-turbo", "qwen-max"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
}

# The main experiment intentionally has no additional system instruction.
# The scale instruction and items are sent as the user message; the workbook's
# A1 title remains local metadata and is not included in the model prompt.
SYSTEM_PROMPT = ""


def load_user_config() -> dict:
    with _CONFIG_LOCK:
        return _load_user_config()


def _load_user_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
    else:
        cfg = {}
    if not isinstance(cfg, dict):
        return {}
    # 配置中的目录一律改写为相对路径，避免拷贝后仍指向旧盘符
    if cfg:
        changed = False
        for key in ("data_dir", "results_dir"):
            if key in cfg and cfg[key]:
                new_val = to_rel_path(cfg[key])
                if new_val != cfg[key]:
                    cfg[key] = new_val
                    changed = True
        # 若从旧绝对路径纠正过来，写回磁盘，保证分发包内配置可迁移
        if changed:
            try:
                _write_user_config(cfg)
            except OSError:
                pass
    return cfg


def save_user_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("配置必须是 JSON 对象")
    cfg = dict(cfg)
    with _CONFIG_LOCK:
        for key in ("data_dir", "results_dir"):
            if key in cfg and cfg[key]:
                cfg[key] = to_rel_path(cfg[key])
        _write_user_config(cfg)


def _write_user_config(cfg: dict) -> None:
    payload = json.dumps(cfg, ensure_ascii=False, indent=2)
    with tempfile.NamedTemporaryFile(
        dir=CONFIG_PATH.parent, suffix=".tmp", mode="w", encoding="utf-8", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
        except Exception:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(CONFIG_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def get_api_credentials(provider: str) -> dict:
    """合并默认密钥与用户覆盖配置。"""
    base = dict(DEFAULT_API_KEYS.get(provider, {}))
    keys = load_user_config().get("api_keys") or {}
    user = keys.get(provider, {}) if isinstance(keys, dict) else {}
    if isinstance(user, dict):
        base.update({k: v for k, v in user.items() if v})
    return base


def sanitize_name(name: str) -> str:
    """文件系统安全名称（跨 Windows / macOS）。"""
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in name).strip().rstrip(".")
    return out or "unnamed"


def resolve_path(path: str | Path) -> Path:
    """
    解析路径：相对路径（如 ./data、./results）相对于程序根目录；
    绝对路径解析为真实路径后供读写使用。
    """
    p = Path(path)
    if p.is_absolute():
        try:
            return p.resolve()
        except OSError:
            return p
    return (ROOT / p).resolve()


def to_rel_path(path: str | Path) -> str:
    """同盘路径相对保存；跨盘目录保留原位置，避免误写到同名目录。"""
    raw = str(path).strip()
    if not raw:
        return raw
    p = Path(raw)
    try:
        root = ROOT.resolve()
        resolved = p.resolve() if p.is_absolute() else (ROOT / p).resolve()
        try:
            rel = resolved.relative_to(root)
            out = rel.as_posix()
        except ValueError:
            try:
                out = os.path.relpath(str(resolved), str(root)).replace("\\", "/")
            except ValueError:
                # Windows cannot represent a different drive as a relative path.
                return resolved.as_posix()
        if out in ("", "."):
            return "./"
        if not out.startswith("."):
            out = "./" + out
        return out
    except (OSError, ValueError):
        return raw.replace("\\", "/")


def display_path(path: str | Path) -> str:
    """日志/界面展示用相对路径。"""
    return to_rel_path(path)
