"""路径与默认配置：相对工程根目录，打包后仍可迁移。"""

from __future__ import annotations

import json
import os
import sys
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

DEFAULT_DATA_DIR = "./data"
DEFAULT_RESULTS_DIR = "./results"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CALL_COUNT = 100
DEFAULT_CONCURRENCY = 5
DEFAULT_MAX_RETRIES = 10
DEFAULT_RETRY_DELAY_SEC = 2.0

# 默认端点；API 密钥只从被忽略的 user_config.json 或界面配置读取。
DEFAULT_API_KEYS = {
    "deepseek": {"api_key": "", "base_url": "https://api.deepseek.com"},
    "openai": {"api_key": "", "base_url": "https://api.openai.com/v1"},
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

SYSTEM_PROMPT = "You are a helpful assistant"


def load_user_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    else:
        cfg = {}
    if isinstance(cfg, dict):
        changed = False
        for key in ("data_dir", "results_dir"):
            if key in cfg and cfg[key]:
                new_val = to_rel_path(cfg[key])
                if new_val != cfg[key]:
                    cfg[key] = new_val
                    changed = True
        if changed:
            try:
                CONFIG_PATH.write_text(
                    json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError:
                pass
    return cfg


def save_user_config(cfg: dict) -> None:
    if isinstance(cfg, dict):
        for key in ("data_dir", "results_dir"):
            if key in cfg and cfg[key]:
                cfg[key] = to_rel_path(cfg[key])
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_api_credentials(provider: str) -> dict:
    base = dict(DEFAULT_API_KEYS.get(provider, {}))
    user = load_user_config().get("api_keys", {}).get(provider, {})
    if isinstance(user, dict):
        base.update({k: v for k, v in user.items() if v})
    return base


def sanitize_name(name: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in name).strip().rstrip(".")
    return out or "unnamed"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        try:
            return p.resolve()
        except OSError:
            return p
    return (ROOT / p).resolve()


def to_rel_path(path: str | Path) -> str:
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
            out = os.path.relpath(str(resolved), str(root)).replace("\\", "/")
        if out in ("", "."):
            return "./"
        if not out.startswith("."):
            out = "./" + out
        return out
    except Exception:
        s = raw.replace("\\", "/")
        if len(s) >= 2 and s[1] == ":":
            return "./" + (Path(s).name or "data")
        if not s.startswith("."):
            s = "./" + s.lstrip("/")
        return s


def display_path(path: str | Path) -> str:
    return to_rel_path(path)
