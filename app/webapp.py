"""大语言模型潜在文化倾向性研究 — Web 前端服务（对接现有后端）。"""

from __future__ import annotations

import queue
import threading
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, render_template, request, send_from_directory, stream_with_context

from . import proxy_util
from .api_client import list_models, stream_chat
from .config import (
    DEFAULT_CALL_COUNT,
    DEFAULT_CONCURRENCY,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TEMPERATURE,
    LOGS_DIR,
    ROOT,
    SYSTEM_PROMPT,
    get_api_credentials,
    load_user_config,
    resolve_path,
    save_user_config,
    to_rel_path,
)
from .net_info import detect_public_ip_info, network_status_text
from .runner import BatchRunner, RunnerConfig, ScaleRunConfig
from .scale_loader import discover_scales

APP_NAME = "大语言模型潜在文化倾向性研究"
APP_SUBTITLE = "LLM Latent Cultural Orientation Research"


def _resource_base() -> Path:
    """开发环境用源码目录；PyInstaller 打包后用解压资源目录。"""
    import sys

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "app"
    return Path(__file__).resolve().parent


_BASE = _resource_base()
app = Flask(
    __name__,
    template_folder=str(_BASE / "templates"),
    static_folder=str(_BASE / "static"),
    static_url_path="/static",
)

_state_lock = threading.Lock()
_runner: Optional[BatchRunner] = None
_worker: Optional[threading.Thread] = None
_log_queue: queue.Queue = queue.Queue()
_progress = {"done": 0, "total": 0, "running": False, "message": ""}


def _push_log(msg: str) -> None:
    _log_queue.put(msg)
    with _state_lock:
        _progress["message"] = msg


def _set_progress(done: int, total: int) -> None:
    with _state_lock:
        _progress["done"] = done
        _progress["total"] = total


def _is_running() -> bool:
    with _state_lock:
        return bool(_progress["running"])


# ------------------------------------------------------------------ pages
@app.get("/")
def welcome_page():
    return render_template(
        "welcome.html",
        app_name=APP_NAME,
        app_subtitle=APP_SUBTITLE,
    )


@app.get("/console")
def console_page():
    return render_template(
        "console.html",
        app_name=APP_NAME,
        app_subtitle=APP_SUBTITLE,
    )


@app.get("/chat")
def chat_page():
    return render_template(
        "chat.html",
        app_name=APP_NAME,
        app_subtitle=APP_SUBTITLE,
    )


@app.get("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "bg.jpg")


# ------------------------------------------------------------------ APIs
@app.get("/api/meta")
def api_meta():
    return jsonify(
        {
            "app_name": APP_NAME,
            "app_subtitle": APP_SUBTITLE,
            "root": ".",
        }
    )


@app.get("/api/network")
def api_network():
    proxy_util.apply_proxy_to_env()
    proxy = proxy_util.detect_system_proxy()
    try:
        info = detect_public_ip_info()
    except Exception as exc:
        info = {"ip": None, "location": f"查询失败: {exc}", "org": ""}
    return jsonify(
        {
            "ip": info.get("ip"),
            "location": info.get("location"),
            "org": info.get("org") or "",
            "proxy": proxy,
            "text": network_status_text(info, proxy),
        }
    )


@app.get("/api/config")
def api_get_config():
    cfg = load_user_config()
    return jsonify(
        {
            "data_dir": cfg.get("data_dir", DEFAULT_DATA_DIR),
            "results_dir": cfg.get("results_dir", DEFAULT_RESULTS_DIR),
            "provider": cfg.get("provider", "deepseek"),
            "model": cfg.get("model", ""),
            "temperature": cfg.get("temperature", DEFAULT_TEMPERATURE),
            "concurrency": cfg.get("concurrency", DEFAULT_CONCURRENCY),
            "batch_en": cfg.get("batch_en", DEFAULT_CALL_COUNT),
            "batch_ch": cfg.get("batch_ch", DEFAULT_CALL_COUNT),
            "scale_vars": cfg.get("scale_vars", {}),
        }
    )


@app.post("/api/config")
def api_save_config():
    data = request.get_json(force=True, silent=True) or {}
    cfg = load_user_config()
    for key in (
        "data_dir",
        "results_dir",
        "provider",
        "model",
        "temperature",
        "concurrency",
        "batch_en",
        "batch_ch",
        "scale_vars",
    ):
        if key in data:
            cfg[key] = data[key]
    # 路径尽量存成相对路径，方便整夹拷贝
    if "data_dir" in cfg:
        cfg["data_dir"] = to_rel_path(cfg["data_dir"])
    if "results_dir" in cfg:
        cfg["results_dir"] = to_rel_path(cfg["results_dir"])
    save_user_config(cfg)
    return jsonify({"ok": True, "data_dir": cfg.get("data_dir"), "results_dir": cfg.get("results_dir")})


@app.get("/api/keys")
def api_get_keys():
    out = {}
    for p in ("deepseek", "openai", "qwen", "gemini"):
        cred = get_api_credentials(p)
        key = cred.get("api_key", "")
        masked = (key[:8] + "…" + key[-4:]) if len(key) > 12 else ("*" * len(key) if key else "")
        out[p] = {
            "api_key": key,
            "api_key_masked": masked,
            "base_url": cred.get("base_url", ""),
        }
    return jsonify(out)


@app.post("/api/keys")
def api_save_keys():
    data = request.get_json(force=True, silent=True) or {}
    cfg = load_user_config()
    existing = dict(cfg.get("api_keys") or {})
    for p in ("deepseek", "openai", "qwen", "gemini"):
        if p in data and isinstance(data[p], dict):
            existing[p] = {
                "api_key": str(data[p].get("api_key", "")).strip(),
                "base_url": str(data[p].get("base_url", "")).strip(),
            }
    cfg["api_keys"] = existing
    save_user_config(cfg)
    return jsonify({"ok": True})


@app.get("/api/scales")
def api_scales():
    data_dir = request.args.get("data_dir") or load_user_config().get(
        "data_dir", DEFAULT_DATA_DIR
    )
    path = resolve_path(data_dir)
    scales = discover_scales(path)
    cfg = load_user_config()
    scale_vars = cfg.get("scale_vars") or {}
    valid_providers = set(DEFAULT_MODELS.keys())

    def _norm_provider(value: str, fallback: str) -> str:
        p = str(value or "").strip()
        if p in valid_providers:
            return p
        fb = str(fallback or "").strip()
        return fb if fb in valid_providers else "deepseek"

    items = []
    for sc in scales:
        prev = scale_vars.get(sc.name, {})
        default_p = _norm_provider(
            prev.get("provider") or cfg.get("provider") or "deepseek",
            "deepseek",
        )
        default_m = prev.get("model") or cfg.get("model") or ""
        en_p = _norm_provider(prev.get("en_provider") or default_p, default_p)
        ch_p = _norm_provider(prev.get("ch_provider") or default_p, default_p)
        items.append(
            {
                "name": sc.name,
                "en_n": sc.en.n_items if sc.en else 0,
                "ch_n": sc.ch.n_items if sc.ch else 0,
                "enabled": prev.get("enabled", True),
                "en_provider": en_p,
                "en_model": prev.get("en_model") or default_m,
                "ch_provider": ch_p,
                "ch_model": prev.get("ch_model") or default_m,
                "en_count": prev.get("en_count", cfg.get("batch_en", DEFAULT_CALL_COUNT)),
                "ch_count": prev.get("ch_count", cfg.get("batch_ch", DEFAULT_CALL_COUNT)),
            }
        )
    return jsonify({"data_dir": to_rel_path(data_dir), "resolved": to_rel_path(path), "scales": items})


@app.get("/api/models")
def api_models():
    provider = request.args.get("provider", "deepseek")
    try:
        cred = get_api_credentials(provider)
        models = list_models(provider, cred.get("api_key"), cred.get("base_url"))
        if not models:
            models = list(DEFAULT_MODELS.get(provider, []))
    except Exception:
        models = list(DEFAULT_MODELS.get(provider, []))
    return jsonify({"provider": provider, "models": models})


@app.post("/api/chat/stream")
def api_chat_stream():
    """SSE streaming chat for the model dialogue test page."""
    data = request.get_json(force=True, silent=True) or {}
    provider = str(data.get("provider") or "deepseek").strip()
    model = str(data.get("model") or "").strip()
    messages = data.get("messages") or []
    try:
        temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
    except (TypeError, ValueError):
        temperature = DEFAULT_TEMPERATURE
    if not model:
        return jsonify({"ok": False, "error": "请选择模型"}), 400
    if not isinstance(messages, list) or not messages:
        return jsonify({"ok": False, "error": "消息不能为空"}), 400

    cred = get_api_credentials(provider)
    proxy_util.apply_proxy_to_env()

    def event_stream():
        import json as _json

        try:
            yield f"data: {_json.dumps({'type': 'start', 'provider': provider, 'model': model}, ensure_ascii=False)}\n\n"
            for chunk in stream_chat(
                provider,
                messages,
                model=model,
                temperature=temperature,
                api_key=cred.get("api_key"),
                base_url=cred.get("base_url"),
                system_prompt=SYSTEM_PROMPT,
            ):
                yield f"data: {_json.dumps({'type': 'delta', 'content': chunk}, ensure_ascii=False)}\n\n"
            yield f"data: {_json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {_json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )


@app.get("/api/run/status")
def api_run_status():
    with _state_lock:
        return jsonify(dict(_progress))


@app.get("/api/run/logs")
def api_run_logs():
    """SSE：实时推送运行日志。"""

    def stream():
        yield "data: {\"type\":\"ready\"}\n\n"
        while True:
            try:
                msg = _log_queue.get(timeout=1.0)
                # escape for JSON string
                safe = (
                    msg.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("\n", "\\n")
                    .replace("\r", "")
                )
                yield f'data: {{"type":"log","message":"{safe}"}}\n\n'
            except queue.Empty:
                with _state_lock:
                    running = _progress["running"]
                    done = _progress["done"]
                    total = _progress["total"]
                yield f'data: {{"type":"ping","running":{str(running).lower()},"done":{done},"total":{total}}}\n\n'
                if not running and _log_queue.empty():
                    # keep connection briefly after finish
                    continue

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/run/start")
def api_run_start():
    global _runner, _worker
    if _is_running():
        return jsonify({"ok": False, "error": "任务正在运行中"}), 400

    data = request.get_json(force=True, silent=True) or {}
    # 先保存配置
    cfg = load_user_config()
    for key in (
        "data_dir",
        "results_dir",
        "provider",
        "model",
        "temperature",
        "concurrency",
        "batch_en",
        "batch_ch",
        "scale_vars",
    ):
        if key in data:
            cfg[key] = data[key]
    save_user_config(cfg)

    data_dir = resolve_path(cfg.get("data_dir", DEFAULT_DATA_DIR))
    scales = discover_scales(data_dir)
    if not scales:
        return jsonify({"ok": False, "error": "未找到量表，请检查数据目录"}), 400

    scale_vars: Dict[str, Any] = cfg.get("scale_vars") or {}
    scale_cfgs: Dict[str, ScaleRunConfig] = {}
    missing = []
    default_p = str(cfg.get("provider") or "deepseek")
    default_m = str(cfg.get("model") or "").strip()
    for sc in scales:
        meta = scale_vars.get(sc.name, {})
        enabled = bool(meta.get("enabled", True))
        legacy_p = str(meta.get("provider") or default_p)
        legacy_m = str(meta.get("model") or default_m).strip()
        en_provider = str(meta.get("en_provider") or legacy_p).strip()
        en_model = str(meta.get("en_model") or legacy_m).strip()
        ch_provider = str(meta.get("ch_provider") or legacy_p).strip()
        ch_model = str(meta.get("ch_model") or legacy_m).strip()
        en_count = int(meta.get("en_count", cfg.get("batch_en", DEFAULT_CALL_COUNT)))
        ch_count = int(meta.get("ch_count", cfg.get("batch_ch", DEFAULT_CALL_COUNT)))
        if enabled:
            if sc.en and en_count > 0 and (not en_provider or not en_model):
                missing.append(f"{sc.name}/en")
            if sc.ch and ch_count > 0 and (not ch_provider or not ch_model):
                missing.append(f"{sc.name}/ch")
        scale_cfgs[sc.name] = ScaleRunConfig(
            scale_name=sc.name,
            en_count=en_count,
            ch_count=ch_count,
            enabled=enabled,
            en_provider=en_provider,
            en_model=en_model,
            ch_provider=ch_provider,
            ch_model=ch_model,
            provider=legacy_p,
            model=legacy_m,
        )
    if missing:
        return jsonify(
            {"ok": False, "error": "以下量表语言未指定 API/模型: " + ", ".join(missing)}
        ), 400

    try:
        concurrency = int(cfg.get("concurrency", DEFAULT_CONCURRENCY))
    except (TypeError, ValueError):
        concurrency = DEFAULT_CONCURRENCY
    if concurrency < 1:
        return jsonify({"ok": False, "error": "并发数至少为 1"}), 400

    # 清空旧日志队列
    while not _log_queue.empty():
        try:
            _log_queue.get_nowait()
        except queue.Empty:
            break

    with _state_lock:
        _progress.update({"done": 0, "total": 0, "running": True, "message": "启动中…"})

    proxy_util.apply_proxy_to_env()
    config = RunnerConfig(
        temperature=float(cfg.get("temperature", DEFAULT_TEMPERATURE)),
        results_root=resolve_path(cfg.get("results_dir", DEFAULT_RESULTS_DIR)),
        logs_root=LOGS_DIR,
        concurrency=concurrency,
        default_provider=str(cfg.get("provider") or "deepseek"),
        default_model=str(cfg.get("model") or "").strip(),
        scale_cfgs=scale_cfgs,
    )

    def on_progress(key: str, done: int, total: int) -> None:
        if key == "__all__":
            _set_progress(done, total)

    _runner = BatchRunner(
        scales=scales,
        config=config,
        log=_push_log,
        progress=on_progress,
    )

    def work() -> None:
        global _runner
        try:
            assert _runner is not None
            _runner.run()
        except Exception as exc:
            _push_log(f"[错误] {exc}")
        finally:
            with _state_lock:
                _progress["running"] = False
            _push_log("运行结束。")

    _worker = threading.Thread(target=work, daemon=True)
    _worker.start()
    _push_log(f"任务已启动（并发={concurrency}）…")
    return jsonify({"ok": True})


@app.post("/api/run/stop")
def api_run_stop():
    if _runner:
        _runner.request_stop()
        _push_log("正在请求停止…")
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "当前无运行任务"}), 400


@app.post("/api/pick-folder")
def api_pick_folder():
    """弹出系统原生文件夹选择对话框。"""
    data = request.get_json(force=True, silent=True) or {}
    kind = data.get("kind", "data")  # data | results
    cfg = load_user_config()
    if kind == "results":
        initial = resolve_path(cfg.get("results_dir", DEFAULT_RESULTS_DIR))
        title = "选择结果保存目录"
    else:
        initial = resolve_path(cfg.get("data_dir", DEFAULT_DATA_DIR))
        title = "选择量表数据目录"
    if not initial.exists():
        initial = ROOT

    chosen = _native_pick_folder(str(initial), title)
    if chosen is None:
        return jsonify({"ok": False, "cancelled": True, "error": "已取消选择"}), 400
    if chosen.startswith("ERROR:"):
        return jsonify({"ok": False, "error": chosen[6:]}), 500
    # 返回相对路径，前端与配置文件均不写死盘符
    rel = to_rel_path(chosen)
    return jsonify({"ok": True, "path": rel})


def _native_pick_folder(initial: str, title: str) -> Optional[str]:
    """优先 PowerShell 系统对话框（便携环境无 tkinter 也能用），再回退 tkinter。"""
    import platform
    import subprocess

    system = platform.system()
    if system == "Windows":
        # FolderBrowserDialog，不依赖 tkinter
        ps = f"""
Add-Type -AssemblyName System.Windows.Forms
$b = New-Object System.Windows.Forms.FolderBrowserDialog
$b.Description = '{title.replace("'", "''")}'
$b.SelectedPath = '{str(initial).replace("'", "''")}'
$b.ShowNewFolderButton = $true
$r = $b.ShowDialog()
if ($r -eq [System.Windows.Forms.DialogResult]::OK) {{
  Write-Output $b.SelectedPath
}}
"""
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", ps],
                capture_output=True,
                text=True,
                timeout=300,
                encoding="utf-8",
                errors="replace",
            )
            path = (completed.stdout or "").strip()
            if path:
                return path
            if completed.returncode != 0 and completed.stderr:
                return "ERROR:" + completed.stderr.strip()
            return None
        except Exception as exc:
            return "ERROR:" + str(exc)

    if system == "Darwin":
        script = (
            f'POSIX path of (choose folder with prompt "{title}" '
            f'default location POSIX file "{initial}")'
        )
        try:
            completed = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=300,
            )
            path = (completed.stdout or "").strip()
            return path or None
        except Exception as exc:
            return "ERROR:" + str(exc)

    # 其他平台尝试 tkinter
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        chosen = filedialog.askdirectory(
            parent=root, initialdir=initial, title=title, mustexist=True
        )
        root.destroy()
        return chosen or None
    except Exception as exc:
        return "ERROR:" + str(exc)


@app.post("/api/open-folder")
def api_open_folder():
    data = request.get_json(force=True, silent=True) or {}
    kind = data.get("kind", "results")
    if kind == "logs":
        path = LOGS_DIR
    else:
        cfg = load_user_config()
        path = resolve_path(cfg.get("results_dir", DEFAULT_RESULTS_DIR))
    path.mkdir(parents=True, exist_ok=True)
    import os
    import platform
    import subprocess

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return jsonify({"ok": True, "path": to_rel_path(path)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def _find_free_port(host: str, start: int = 8765, span: int = 20) -> int:
    import socket

    for port in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    return start


def _open_browser(url: str) -> None:
    """从 bat/双击启动时，Windows 用 start 更可靠。"""
    import os
    import platform
    import subprocess
    import time
    import urllib.request

    # 等服务真正可访问再打开，避免“连不上就关了”
    for _ in range(40):
        try:
            with urllib.request.urlopen(url, timeout=0.5) as resp:
                if resp.status < 500:
                    break
        except Exception:
            time.sleep(0.25)
    else:
        print(f"[警告] 服务似乎尚未就绪，仍尝试打开: {url}")

    system = platform.system()
    try:
        if system == "Windows":
            # start 会调用系统默认浏览器；空标题参数不可省略
            subprocess.Popen(
                ["cmd", "/c", "start", "", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif system == "Darwin":
            subprocess.Popen(["open", url])
        else:
            subprocess.Popen(["xdg-open", url])
        print(f"已请求打开浏览器: {url}")
        return
    except Exception as exc:
        print(f"[警告] 系统打开浏览器失败: {exc}，尝试 webbrowser…")

    try:
        webbrowser.open(url)
        print(f"已通过 webbrowser 打开: {url}")
    except Exception as exc:
        print(f"[错误] 无法自动打开浏览器: {exc}")
        print(f"请手动在浏览器访问: {url}")


def run_web(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    port = _find_free_port(host, port)
    url = f"http://{host}:{port}/"
    print("=" * 60)
    print(f"  {APP_NAME}")
    print(f"  本地地址: {url}")
    print("  请勿关闭本黑色窗口，关闭即停止服务。")
    print("=" * 60)
    if open_browser:
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()
    # 允许局域网访问可改为 host="0.0.0.0"；默认本机更安全
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
