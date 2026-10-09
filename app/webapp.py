"""大语言模型潜在文化倾向性研究 — Web 前端服务（对接现有后端）。"""

from __future__ import annotations

import queue
import random
import threading
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from flask import Flask, Response, jsonify, render_template, request, send_file, send_from_directory, stream_with_context

from . import proxy_util
from . import __version__ as APP_VERSION
from .startup import application_identity, find_existing_server
from .api_client import list_models, stream_chat
from .generation_config import GenerationConfig, simple_platform_generation_config
from .config import (
    DEFAULT_CAPTURE_MODEL_CATALOG,
    DEFAULT_CAPTURE_NETWORK,
    DEFAULT_CALL_COUNT,
    DEFAULT_CONCURRENCY,
    DEFAULT_SCALE_CONCURRENCY,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RESULTS_DIR,
    DEFAULT_NETWORK_GUARD,
    DEFAULT_TEMPERATURE,
    DEFAULT_CONDITION_MODE,
    LOGS_DIR,
    MAX_CONCURRENCY,
    MAX_SCALE_CONCURRENCY,
    ROOT,
    SYSTEM_PROMPT,
    get_api_credentials,
    load_user_config,
    resolve_path,
    save_user_config,
    to_rel_path,
)
from .net_info import detect_public_ip_info, network_status_text
from .provenance import sanitize_proxy
from .experiment_presets import (
    PRESET_SCHEMA_VERSION,
    apply_preset,
    get_preset,
    list_presets,
    normalize_prompt_config,
    parameter_guide,
)
from .condition_design import (
    CONDITION_SCHEMA_VERSION,
    build_condition_descriptor,
    compare_condition_configs,
    normalize_condition_mode,
)
from .runner import (
    BatchRunner,
    RunnerConfig,
    ScaleRunConfig,
    normalize_sample_mode,
)
from .scale_loader import discover_scales
from .prompting import build_trial_prompt, prompt_preview_sections
from .trial_results import EXPORT_NAME, collect_trials, export_trials, read_trial_detail
from .scale_profiles import (
    KOHLBERG_FORM_OPTIONS,
    TYPED_FLAT_SPECS,
    kohlberg_standalone_form,
    normalize_kohlberg_form,
    select_kohlberg_form,
)

APP_NAME = "大语言模型潜在文化倾向性研究"
APP_SUBTITLE = "LLM Latent Cultural Orientation Research"
SHUFFLE_POLICY_VERSION = "scale_rules_v2"


def _resolved_condition_mode(cfg: Mapping[str, Any]) -> str:
    """Read the explicit mode, migrating older preset-only configurations."""

    raw = cfg.get("condition_mode")
    if raw in (None, ""):
        preset_id = str(cfg.get("preset_id") or "").strip()
        preset = get_preset(preset_id) if preset_id else None
        raw = (preset or {}).get("condition_mode", DEFAULT_CONDITION_MODE)
    return normalize_condition_mode(raw)


def _resolved_order_strategy(cfg: Mapping[str, Any]) -> str:
    """The workbench always applies the confirmed rules for each scale."""

    return "per_scale"


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
# This is a local research workbench. Keeping templates reloadable prevents a new
# JavaScript file from being paired with an old cached table header during edits.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
app.jinja_env.globals["app_version"] = APP_VERSION

_state_lock = threading.Lock()
_start_lock = threading.Lock()
_runner: Optional[BatchRunner] = None
_worker: Optional[threading.Thread] = None
_log_queue: queue.Queue = queue.Queue()
_progress = {"done": 0, "total": 0, "running": False, "message": ""}


_CONFIG_FIELDS = (
    "data_dir", "results_dir", "provider", "model", "preset_id",
    "condition_mode", "prompt_config", "temperature", "generation_config",
    "concurrency", "scale_concurrency", "batch_en", "batch_ch", "scale_vars",
    "strict_capabilities", "capture_network", "capture_model_catalog",
    "network_guard", "max_retries", "max_group_attempts", "sample_mode", "random_seed",
)


@app.before_request
def _validate_json_request():
    if request.method == "POST" and request.path.startswith("/api/"):
        if request.get_data() and not isinstance(request.get_json(silent=True), dict):
            return jsonify({"ok": False, "error": "请求内容必须是 JSON 对象"}), 400


def _generation_from_config(cfg: Mapping[str, Any]) -> GenerationConfig:
    temperature = cfg.get("temperature")
    if temperature in (None, ""):
        temperature = DEFAULT_TEMPERATURE
    return simple_platform_generation_config(
        cfg.get("generation_config"), fallback_temperature=float(temperature)
    )


def _integer_setting(value: Any, label: str, minimum: int = 0, maximum: Optional[int] = None) -> int:
    message = f"{label}必须是至少为 {minimum} 的整数"
    if maximum is not None:
        message = f"{label}必须是 {minimum} 到 {maximum} 之间的整数"
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if (isinstance(value, bool) or (isinstance(value, float) and value != number)
            or number < minimum or (maximum is not None and number > maximum)):
        raise ValueError(message)
    return number


def _trial_count(value: Any) -> int:
    return _integer_setting(value, "运行次数")


def _normalize_execution_config(cfg: Dict[str, Any]) -> None:
    """Use the same integer validation when saving and starting a run."""
    settings = (
        ("concurrency", "单量表 API 并发数", DEFAULT_CONCURRENCY, 1, MAX_CONCURRENCY),
        ("scale_concurrency", "量表并行数", DEFAULT_SCALE_CONCURRENCY, 1, MAX_SCALE_CONCURRENCY),
        ("batch_en", "英文运行次数", DEFAULT_CALL_COUNT, 0, None),
        ("batch_ch", "中文运行次数", DEFAULT_CALL_COUNT, 0, None),
        ("max_retries", "请求尝试次数", DEFAULT_MAX_RETRIES, 1, None),
        ("max_group_attempts", "旧版尝试上限", 0, 0, None),
    )
    for key, label, default, minimum, maximum in settings:
        cfg[key] = _integer_setting(cfg.get(key, default), label, minimum, maximum)
    seed = cfg.get("random_seed")
    cfg["random_seed"] = None if seed in (None, "") else _integer_setting(seed, "随机种子")
    scale_vars = cfg.get("scale_vars", {})
    if not isinstance(scale_vars, Mapping):
        raise ValueError("量表设置必须是 JSON 对象")
    for name, meta in scale_vars.items():
        if not isinstance(meta, dict):
            raise ValueError(f"{name} 的量表设置必须是 JSON 对象")
        for key in ("en_count", "ch_count"):
            if key in meta:
                meta[key] = _trial_count(meta[key])
        meta.pop("shuffle_items", None)
    cfg["scale_vars"] = scale_vars


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse JSON/UI booleans without treating the string ``"false"`` as true."""

    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "on", "是", "开启"}:
        return True
    if text in {"false", "0", "no", "off", "否", "关闭"}:
        return False
    return bool(value)


def _scale_shuffle_enabled(sheet) -> bool:
    """Ignore obsolete switches; validated scale profiles determine ordering."""

    return bool(sheet.shuffle_supported) if sheet is not None else True


def _shuffle_status(sheet) -> str:
    if sheet is not None and not sheet.shuffle_supported:
        return "固定顺序"
    if sheet is not None and sheet.profile_id in {"attribution_bias_v1", "intuitive_reasoning_v1"}:
        return "限定范围乱序"
    return "整行乱序"


def _architecture_meta(sheet) -> Dict[str, Any]:
    """Return user-visible, profile-specific structure for the workbench.

    This is deliberately supplied by the backend rather than inferred from
    item counts in JavaScript.  The table should make it immediately visible
    that a special instrument is being handled by a dedicated parser.
    """

    if sheet is None:
        return {
            "kind": "缺少语言版本",
            "summary": "未加载该语言工作表",
            "detail": "",
            "ready": False,
            "parser": "",
            "shuffle_rule": "",
        }

    profile_id = sheet.profile_id
    special = profile_id != "default_flat_v1"
    typed_flat = profile_id in TYPED_FLAT_SPECS
    if profile_id == "attribution_bias_v1":
        summary = "6 个情境组：A 的 10 个 1–7 评分 + B/C 两个 0–100% 概率题；末尾 6 个群体性题"
        shuffle_rule = "默认将 6 个情境组整体乱序；群体性题固定在末尾"
    elif profile_id == "human_rights_v1":
        summary = "10 个情境 × 观点／结果／行动 3 节 × 每节 2 题（1–5）"
        shuffle_rule = "默认将情境整组乱序；情境内三节和题目顺序固定"
    elif profile_id == "ios_v1":
        summary = "4 个关系题：从图片中的 7 个圆圈对图示中各选一个编号；Excel DISPIMG 图片随请求发送"
        shuffle_rule = "默认将 4 个关系题乱序；图片内图示固定不变"
    elif profile_id == "intuitive_reasoning_v1":
        summary = "40 题：先完成 24 个逻辑有效性判断，再完成 16 个结论可信度评价"
        shuffle_rule = "整份量表一次发送；两个任务顺序固定，各自在任务内部乱序"
    elif profile_id == "kohlberg_mji_v1":
        single_form = kohlberg_standalone_form(sheet)
        if single_form:
            summary = f"Form {single_form}：{sheet.n_items} 个开放式情境"
            shuffle_rule = "访谈保持原顺序，不允许乱序；整份量表作为一个试次发送"
        else:
            summary = "9 个开放式道德两难故事；按 Form A/B/C 选择 3 个"
            shuffle_rule = "固定原顺序；不允许乱序；可选择全部故事或单个 Form"
    elif typed_flat:
        summary = f"{sheet.profile_label}；按量表原有题目作答"
        shuffle_rule = (
            "固定顺序；不允许乱序"
            if not sheet.shuffle_supported
            else "默认乱序作答题项；说明不参与作答"
        )
    else:
        summary = "A1 标题／A2 说明／A3 起平铺题项"
        shuffle_rule = "默认平铺题项乱序；说明不参与作答"

    detail = (
        f"{'类型化量表' if typed_flat else '专属量表'}：{sheet.profile_label}；"
        f"当前题目数：{sheet.n_items}。"
        if special
        else f"通用平铺量表；当前题目数：{sheet.n_items}。"
    )
    if sheet.profile_error:
        detail = f"结构校验未通过：{sheet.profile_error}"
    return {
        "kind": "类型化架构" if typed_flat else ("专属架构" if special else "通用架构"),
        "summary": summary,
        "detail": detail,
        "ready": not bool(sheet.profile_error),
        "parser": "按原文发送；回答保留原文",
        "shuffle_rule": shuffle_rule,
    }


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
        max_concurrency=MAX_CONCURRENCY,
        max_scale_concurrency=MAX_SCALE_CONCURRENCY,
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
            "version": APP_VERSION,
            "root": ".",
        }
    )


@app.get("/api/health")
def api_health():
    import os
    return jsonify({**application_identity(), "pid": os.getpid()})


@app.get("/api/network")
def api_network():
    proxy_util.apply_proxy_to_env()
    proxy = proxy_util.detect_system_proxy()
    proxy_identity = sanitize_proxy(proxy)
    try:
        info = detect_public_ip_info()
    except Exception as exc:
        info = {"ip": None, "location": f"查询失败: {exc}", "org": ""}
    return jsonify(
        {
            "ip": info.get("ip"),
            "location": info.get("location"),
            "org": info.get("org") or "",
            "proxy": proxy_identity.get("display"),
            "proxy_configured": proxy_identity.get("configured", False),
            "proxy_credentials_present": proxy_identity.get(
                "credentials_present", False
            ),
            "text": network_status_text(info, proxy),
        }
    )


@app.get("/api/config")
def api_get_config():
    cfg = load_user_config()
    try:
        condition_mode = _resolved_condition_mode(cfg)
    except ValueError:
        condition_mode = DEFAULT_CONDITION_MODE
    try:
        generation = _generation_from_config(cfg)
        prompt_config = normalize_prompt_config(cfg.get("prompt_config"))
        sample_mode = normalize_sample_mode(cfg.get("sample_mode"))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"配置无效：{exc}"}), 400
    return jsonify(
        {
            "data_dir": cfg.get("data_dir", DEFAULT_DATA_DIR),
            "results_dir": cfg.get("results_dir", DEFAULT_RESULTS_DIR),
            "provider": cfg.get("provider", "deepseek"),
            "model": cfg.get("model", ""),
            "preset_id": cfg.get("preset_id", "custom"),
            "condition_mode": condition_mode,
            "prompt_config": prompt_config,
            "order_strategy": _resolved_order_strategy(cfg),
            "shuffle_policy_version": SHUFFLE_POLICY_VERSION,
            "temperature": generation.temperature,
            "generation_config": generation.to_dict(),
            "strict_capabilities": False,
            "capture_network": _as_bool(
                cfg.get("capture_network"), DEFAULT_CAPTURE_NETWORK
            ),
            "capture_model_catalog": _as_bool(
                cfg.get("capture_model_catalog"), DEFAULT_CAPTURE_MODEL_CATALOG
            ),
            "network_guard": cfg.get("network_guard", DEFAULT_NETWORK_GUARD),
            "max_retries": cfg.get("max_retries", DEFAULT_MAX_RETRIES),
            "max_group_attempts": cfg.get("max_group_attempts", 0),
            "sample_mode": sample_mode,
            "random_seed": cfg.get("random_seed"),
            "concurrency": cfg.get("concurrency", DEFAULT_CONCURRENCY),
            "scale_concurrency": cfg.get("scale_concurrency", DEFAULT_SCALE_CONCURRENCY),
            "batch_en": cfg.get("batch_en", DEFAULT_CALL_COUNT),
            "batch_ch": cfg.get("batch_ch", DEFAULT_CALL_COUNT),
            "scale_vars": cfg.get("scale_vars", {}),
        }
    )


@app.post("/api/config")
def api_save_config():
    data = request.get_json(force=True, silent=True) or {}
    cfg = load_user_config()
    cfg.update({key: data[key] for key in _CONFIG_FIELDS if key in data})
    try:
        simple_generation = _generation_from_config(cfg)
        cfg["generation_config"] = simple_generation.to_dict()
        cfg["temperature"] = simple_generation.temperature
        cfg["prompt_config"] = normalize_prompt_config(cfg.get("prompt_config"))
        _normalize_execution_config(cfg)
    except (TypeError, ValueError, OverflowError) as exc:
        return jsonify({"ok": False, "error": f"配置无效：{exc}"}), 400
    try:
        cfg["sample_mode"] = normalize_sample_mode(cfg.get("sample_mode"))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    preset_id = str(cfg.get("preset_id") or "custom").strip()
    if preset_id != "custom" and get_preset(preset_id) is None:
        return jsonify({"ok": False, "error": f"未知实验预设：{preset_id}"}), 400
    cfg["preset_id"] = preset_id
    cfg["order_strategy"] = _resolved_order_strategy(cfg)
    cfg["shuffle_policy_version"] = SHUFFLE_POLICY_VERSION
    try:
        cfg["condition_mode"] = _resolved_condition_mode(cfg)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    network_guard = str(
        cfg.get("network_guard", DEFAULT_NETWORK_GUARD) or DEFAULT_NETWORK_GUARD
    ).strip().lower()
    if network_guard not in {"warn", "stop"}:
        return jsonify({"ok": False, "error": "network_guard 必须是 warn 或 stop"}), 400
    cfg["network_guard"] = network_guard
    cfg["strict_capabilities"] = False
    cfg["capture_network"] = _as_bool(
        cfg.get("capture_network"), DEFAULT_CAPTURE_NETWORK
    )
    cfg["capture_model_catalog"] = _as_bool(
        cfg.get("capture_model_catalog"), DEFAULT_CAPTURE_MODEL_CATALOG
    )
    # 路径尽量存成相对路径，方便整夹拷贝
    if "data_dir" in cfg:
        cfg["data_dir"] = to_rel_path(cfg["data_dir"])
    if "results_dir" in cfg:
        cfg["results_dir"] = to_rel_path(cfg["results_dir"])
    save_user_config(cfg)
    return jsonify({"ok": True, "data_dir": cfg.get("data_dir"), "results_dir": cfg.get("results_dir")})


@app.get("/api/presets")
def api_get_presets():
    """Return shared experiment presets and the operator-facing parameter guide."""

    return jsonify(
        {
            "schema_version": PRESET_SCHEMA_VERSION,
            "condition_schema_version": CONDITION_SCHEMA_VERSION,
            "presets": list_presets(),
            "parameter_guide": parameter_guide(),
        }
    )


@app.post("/api/presets/apply")
def api_apply_preset():
    """Resolve one preset without changing the persisted user configuration."""

    data = request.get_json(force=True, silent=True) or {}
    try:
        preset = apply_preset(str(data.get("preset_id") or ""))
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "preset": preset})


@app.post("/api/conditions/compare")
def api_compare_conditions():
    """Compare two requested conditions without claiming equal provider defaults."""

    data = request.get_json(force=True, silent=True) or {}
    if not isinstance(data, Mapping):
        return jsonify({"ok": False, "error": "比较条件必须是 JSON 对象"}), 400

    def resolve(value: Any, preset_id: Any) -> tuple[str, GenerationConfig, Dict[str, Any]]:
        chosen = str(preset_id or "").strip()
        if chosen:
            preset = get_preset(chosen)
            if preset is None:
                raise ValueError(f"未知实验预设：{chosen}")
            mode = normalize_condition_mode(preset.get("condition_mode"))
            config = GenerationConfig.from_mapping(
                preset.get("generation_config"), fallback_temperature=1.0
            )
            return mode, config, normalize_prompt_config(preset.get("prompt_config"))
        if not isinstance(value, Mapping):
            raise ValueError("自定义条件必须是 JSON 对象")
        mode = normalize_condition_mode(value.get("condition_mode"))
        config = GenerationConfig.from_mapping(
            value.get("generation_config"), fallback_temperature=1.0
        )
        return mode, config, normalize_prompt_config(value.get("prompt_config"))

    try:
        left_mode, left_config, left_prompt = resolve(
            data.get("left") or {}, data.get("left_preset_id")
        )
        right_mode, right_config, right_prompt = resolve(
            data.get("right") or {}, data.get("right_preset_id")
        )
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    comparison = compare_condition_configs(left_config, right_config)
    prompt_differences = [
        f"prompt_config.{key}"
        for key in sorted(set(left_prompt) | set(right_prompt))
        if left_prompt.get(key) != right_prompt.get(key)
    ]
    comparison["requested_difference_fields"].extend(prompt_differences)
    comparison["requested_equivalent"] = not comparison["requested_difference_fields"]
    comparison.update(
        {
            "left_mode": left_mode,
            "right_mode": right_mode,
            "left_descriptor": build_condition_descriptor(
                left_mode, left_config, prompt_contract=left_prompt["contract"],
                cultural_identity=left_prompt["cultural_identity"],
            ),
            "right_descriptor": build_condition_descriptor(
                right_mode, right_config, prompt_contract=right_prompt["contract"],
                cultural_identity=right_prompt["cultural_identity"],
            ),
        }
    )
    return jsonify({"ok": True, "comparison": comparison})


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
        reference_sheet = sc.en or sc.ch
        profile_errors = {
            language: sheet.profile_error
            for language, sheet in (("en", sc.en), ("ch", sc.ch))
            if sheet is not None and sheet.profile_error
        }
        prev = scale_vars.get(sc.name, {})
        if not isinstance(prev, Mapping):
            prev = {}
        shuffle_supported = any(
            sheet is not None and sheet.shuffle_supported
            for sheet in (sc.en, sc.ch)
        )
        default_p = _norm_provider(
            prev.get("provider") or cfg.get("provider") or "deepseek",
            "deepseek",
        )
        default_m = prev.get("model") or cfg.get("model") or ""
        en_p = _norm_provider(prev.get("en_provider") or default_p, default_p)
        ch_p = _norm_provider(prev.get("ch_provider") or default_p, default_p)
        mji_form = (
            normalize_kohlberg_form(prev.get("mji_form"))
            if reference_sheet
            and reference_sheet.profile_id == "kohlberg_mji_v1"
            and not kohlberg_standalone_form(reference_sheet)
            else None
        )
        items.append(
            {
                "name": sc.name,
                "en_n": sc.en.n_items if sc.en else 0,
                "ch_n": sc.ch.n_items if sc.ch else 0,
                "en_images": len(sc.en.images) if sc.en else 0,
                "ch_images": len(sc.ch.images) if sc.ch else 0,
                "enabled": prev.get("enabled", True),
                "shuffle_items": _scale_shuffle_enabled(reference_sheet),
                "shuffle_status": _shuffle_status(reference_sheet),
                "shuffle_supported": shuffle_supported,
                "profile_id": reference_sheet.profile_id if reference_sheet else None,
                "profile_label": reference_sheet.profile_label if reference_sheet else "",
                "profile_errors": profile_errors,
                "en_architecture": _architecture_meta(sc.en),
                "ch_architecture": _architecture_meta(sc.ch),
                "en_provider": en_p,
                "en_model": prev.get("en_model") or default_m,
                "ch_provider": ch_p,
                "ch_model": prev.get("ch_model") or default_m,
                "en_count": prev.get("en_count", cfg.get("batch_en", DEFAULT_CALL_COUNT)),
                "ch_count": prev.get("ch_count", cfg.get("batch_ch", DEFAULT_CALL_COUNT)),
                "mji_form": mji_form,
                "mji_form_options": (
                    [{"value": value, "label": label} for value, label in KOHLBERG_FORM_OPTIONS.items()]
                    if mji_form is not None
                    else []
                ),
            }
        )
    return jsonify({"data_dir": to_rel_path(data_dir), "resolved": to_rel_path(path), "scales": items})


@app.post("/api/prompt-preview")
def api_prompt_preview():
    data = request.get_json(silent=True) or {}
    cfg = data.get("config") or {}
    if not isinstance(cfg, Mapping):
        return jsonify({"error": "预览配置无效"}), 400
    language = data.get("language", "ch")
    if language not in ("en", "ch"):
        return jsonify({"error": "请选择中文或英文"}), 400
    try:
        source = resolve_path(cfg.get("data_dir") or DEFAULT_DATA_DIR)
        scale = next((item for item in discover_scales(source) if item.name == data.get("scale_name")), None)
        if scale is None:
            raise ValueError("未找到所选量表，请刷新量表列表")
        sheet = scale.ch if language == "ch" else scale.en
        if sheet is None:
            raise ValueError("量表没有所选语言版本")
        if sheet.profile_error:
            raise ValueError(sheet.profile_error)
        scale_vars = cfg.get("scale_vars") or {}
        if not isinstance(scale_vars, Mapping):
            raise ValueError("量表设置必须是 JSON 对象")
        meta = scale_vars.get(scale.name) or {}
        if not isinstance(meta, Mapping):
            raise ValueError("所选量表设置无效")
        if sheet.profile_id == "kohlberg_mji_v1" and not kohlberg_standalone_form(sheet):
            form = normalize_kohlberg_form(meta.get("mji_form"))
            if form != "all":
                sheet = select_kohlberg_form(sheet, form)
        prompt_cfg = normalize_prompt_config(cfg.get("prompt_config"))
        shuffle = _scale_shuffle_enabled(sheet)
        prompt, _order, _plan = build_trial_prompt(
            sheet, language, shuffle, random.Random(),
            prompt_cfg["contract"], prompt_cfg["cultural_identity"],
        )
        return jsonify({
            "prompt": prompt,
            "sections": prompt_preview_sections(sheet, language, prompt, prompt_cfg["contract"], prompt_cfg["cultural_identity"]),
            "images": [{"src": item.data_url, "anchor": item.anchor} for item in sheet.images],
            "note": "示例顺序，正式试次可能不同。" if shuffle else "按量表固定顺序显示。",
        })
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400


def _trial_root(value: Any = None) -> Path:
    cfg = load_user_config()
    return resolve_path(value or cfg.get("results_dir") or DEFAULT_RESULTS_DIR).resolve()


def _trial_folder(root: Path, run: Any) -> Path:
    name = str(run or ".")
    if name != "." and (name in {"", ".."} or "/" in name or "\\" in name):
        raise ValueError("运行目录无效")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError("运行目录不存在")
    return path


@app.get("/api/trials/runs")
def api_trial_runs():
    root = _trial_root(request.args.get("results_dir"))
    runs = []
    if root.is_dir():
        folders = [path for path in root.iterdir() if path.is_dir()
                   and path.resolve().is_relative_to(root)
                   and (path.name.startswith("run_") or (path / "plan.jsonl").is_file())]
        folders.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        runs = [{"id": path.name, "label": path.name} for path in folders]
        if any(root.iterdir()):
            runs.append({"id": ".", "label": "全部运行与旧结果"})
    return jsonify({"runs": runs})


@app.get("/api/trials")
def api_trial_list():
    try:
        folder = _trial_folder(_trial_root(request.args.get("results_dir")), request.args.get("run"))
        rows = collect_trials(folder)
        return jsonify({"trials": [
            {key: value for key, value in row.items() if key not in {"raw_response", "prompt"}}
            for row in rows
        ]})
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/trials/detail")
def api_trial_detail():
    try:
        folder = _trial_folder(_trial_root(request.args.get("results_dir")), request.args.get("run"))
        row = read_trial_detail(folder, request.args.get("source", ""), request.args.get("kind", "result"))
        return jsonify({"raw_response": row["raw_response"], "status_label": row["status_label"]})
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/trials/export")
def api_trial_export():
    data = request.get_json(silent=True) or {}
    try:
        folder = _trial_folder(_trial_root(data.get("results_dir")), data.get("run"))
        path = export_trials(folder)
        return send_file(path, as_attachment=True, download_name=EXPORT_NAME)
    except (OSError, ValueError) as exc:
        return jsonify({"error": f"导出失败：{exc}"}), 400


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

    try:
        generation_config = simple_platform_generation_config(
            data.get("generation_config"), fallback_temperature=temperature
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": f"生成参数无效：{exc}"}), 400

    cred = get_api_credentials(provider)
    proxy_util.apply_proxy_to_env()
    client_request_id = f"chat-{uuid.uuid4().hex}"

    def event_stream():
        import json as _json

        try:
            yield f"data: {_json.dumps({'type': 'start', 'provider': provider, 'model': model, 'client_request_id': client_request_id}, ensure_ascii=False)}\n\n"
            for chunk in stream_chat(
                provider,
                messages,
                model=model,
                temperature=temperature,
                api_key=cred.get("api_key"),
                base_url=cred.get("base_url"),
                system_prompt=SYSTEM_PROMPT,
                generation_config=generation_config,
                client_request_id=client_request_id,
            ):
                yield f"data: {_json.dumps({'type': 'delta', 'content': chunk}, ensure_ascii=False)}\n\n"
            yield f"data: {_json.dumps({'type': 'done', 'client_request_id': client_request_id}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {_json.dumps({'type': 'error', 'message': str(exc), 'client_request_id': client_request_id}, ensure_ascii=False)}\n\n"

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
    with _start_lock:
        try:
            return _start_run()
        except (OSError, RuntimeError, TypeError, ValueError, OverflowError) as exc:
            return jsonify({"ok": False, "error": f"无法启动：{exc}"}), 400


def _start_run():
    global _runner, _worker
    if _is_running():
        return jsonify({"ok": False, "error": "任务正在运行中"}), 400

    data = request.get_json(force=True, silent=True) or {}
    # 先读取本次提交，验证通过后再保存配置；避免失败配置覆盖上一次可运行设置。
    cfg = load_user_config()
    cfg.update({key: data[key] for key in _CONFIG_FIELDS if key in data})
    _normalize_execution_config(cfg)

    data_dir = resolve_path(cfg.get("data_dir", DEFAULT_DATA_DIR))
    scales = discover_scales(data_dir)
    if not scales:
        return jsonify({"ok": False, "error": "未找到量表，请检查数据目录"}), 400

    scale_vars = cfg.get("scale_vars") or {}
    if not isinstance(scale_vars, Mapping):
        raise ValueError("量表设置必须是 JSON 对象")
    order_strategy = _resolved_order_strategy(cfg)
    scale_cfgs: Dict[str, ScaleRunConfig] = {}
    missing = []
    default_p = str(cfg.get("provider") or "deepseek")
    default_m = str(cfg.get("model") or "").strip()
    for sc in scales:
        meta = scale_vars.get(sc.name, {})
        if not isinstance(meta, Mapping):
            meta = {}
        enabled = _as_bool(meta.get("enabled"), True)
        legacy_p = str(meta.get("provider") or default_p)
        legacy_m = str(meta.get("model") or default_m).strip()
        en_provider = str(meta.get("en_provider") or legacy_p).strip()
        en_model = str(meta.get("en_model") or legacy_m).strip()
        ch_provider = str(meta.get("ch_provider") or legacy_p).strip()
        ch_model = str(meta.get("ch_model") or legacy_m).strip()
        en_count = _trial_count(meta.get("en_count", cfg.get("batch_en", DEFAULT_CALL_COUNT)))
        ch_count = _trial_count(meta.get("ch_count", cfg.get("batch_ch", DEFAULT_CALL_COUNT)))
        if enabled:
            if (
                sc.en
                and not sc.en.profile_error
                and en_count > 0
                and (not en_provider or not en_model)
            ):
                missing.append(f"{sc.name}/en")
            if (
                sc.ch
                and not sc.ch.profile_error
                and ch_count > 0
                and (not ch_provider or not ch_model)
            ):
                missing.append(f"{sc.name}/ch")
        shuffle_items = _scale_shuffle_enabled(sc.en or sc.ch)
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
            shuffle_items=shuffle_items,
            mji_form=str(meta.get("mji_form") or "all"),
        )
    if missing:
        return jsonify(
            {"ok": False, "error": "以下量表语言未指定 API/模型: " + ", ".join(missing)}
        ), 400

    concurrency = cfg["concurrency"]
    scale_concurrency = cfg["scale_concurrency"]

    # 清空旧日志队列
    while not _log_queue.empty():
        try:
            _log_queue.get_nowait()
        except queue.Empty:
            break

    proxy_util.apply_proxy_to_env()
    try:
        generation_config = _generation_from_config(cfg)
        prompt_config = normalize_prompt_config(cfg.get("prompt_config"))
        max_retries = cfg["max_retries"]
        max_group_attempts = cfg["max_group_attempts"]
        sample_mode = normalize_sample_mode(cfg.get("sample_mode"))
        preset_id = str(cfg.get("preset_id") or "custom").strip()
        if preset_id != "custom" and get_preset(preset_id) is None:
            raise ValueError(f"未知实验预设：{preset_id}")
        condition_mode = _resolved_condition_mode(cfg)
        random_seed = cfg["random_seed"]
        network_guard = str(
            cfg.get("network_guard", DEFAULT_NETWORK_GUARD) or DEFAULT_NETWORK_GUARD
        ).strip().lower()
        if network_guard not in {"warn", "stop"}:
            raise ValueError("network_guard 必须是 warn 或 stop")
    except (TypeError, ValueError) as exc:
        with _state_lock:
            _progress["running"] = False
        return jsonify({"ok": False, "error": f"生成参数无效：{exc}"}), 400

    config = RunnerConfig(
        temperature=generation_config.temperature,
        results_root=resolve_path(cfg.get("results_dir", DEFAULT_RESULTS_DIR)),
        logs_root=LOGS_DIR,
        concurrency=concurrency,
        scale_concurrency=scale_concurrency,
        default_provider=str(cfg.get("provider") or "deepseek"),
        default_model=str(cfg.get("model") or "").strip(),
        preset_id=preset_id,
        condition_mode=condition_mode,
        prompt_contract=prompt_config["contract"],
        cultural_identity=prompt_config["cultural_identity"],
        order_strategy=order_strategy,
        scale_cfgs=scale_cfgs,
        generation_config=generation_config,
        strict_capabilities=False,
        max_retries=max_retries,
        max_group_attempts=max_group_attempts,
        sample_mode=sample_mode,
        random_seed=random_seed,
        capture_network=_as_bool(
            cfg.get("capture_network"), DEFAULT_CAPTURE_NETWORK
        ),
        capture_model_catalog=_as_bool(
            cfg.get("capture_model_catalog"), DEFAULT_CAPTURE_MODEL_CATALOG
        ),
        network_guard=network_guard,
    )

    def on_progress(key: str, done: int, total: int) -> None:
        if key == "__all__":
            _set_progress(done, total)

    runner = BatchRunner(
        scales=scales,
        config=config,
        log=_push_log,
        progress=on_progress,
    )

    # Run the same capability gate synchronously before persisting config or
    # starting the worker. The worker repeats it defensively.
    capability_errors = runner.validate_formal_configuration()
    if capability_errors:
        with _state_lock:
            _progress["running"] = False
        return jsonify(
            {
                "ok": False,
                "error": "当前模型或参数设置无法运行：" + "；".join(capability_errors),
            }
        ), 400

    cfg["temperature"] = generation_config.temperature
    cfg["generation_config"] = generation_config.to_dict()
    cfg["scale_concurrency"] = scale_concurrency
    cfg["shuffle_policy_version"] = SHUFFLE_POLICY_VERSION
    cfg["order_strategy"] = order_strategy
    cfg["prompt_config"] = normalize_prompt_config(cfg.get("prompt_config"))
    cfg["strict_capabilities"] = config.strict_capabilities
    cfg["capture_network"] = config.capture_network
    cfg["capture_model_catalog"] = config.capture_model_catalog
    cfg["network_guard"] = config.network_guard
    save_user_config(cfg)

    def work() -> None:
        try:
            runner.run()
        except Exception as exc:
            _push_log(f"[错误] {exc}")
        finally:
            with _state_lock:
                _progress["running"] = False
            _push_log("运行结束。")

    _runner = runner
    with _state_lock:
        _progress.update({"done": 0, "total": 0, "running": True, "message": "启动中…"})
    _worker = threading.Thread(target=work, daemon=True)
    try:
        _worker.start()
    except (OSError, RuntimeError):
        _runner = _worker = None
        with _state_lock:
            _progress["running"] = False
        raise
    _push_log(
        f"任务已启动（量表并行={scale_concurrency}，单量表 API 并发={concurrency}）…"
    )
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
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
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
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError("没有可用的本地端口，请关闭旧平台窗口后重试")


def _open_browser(url: str) -> None:
    """从 bat/双击启动时，Windows 用 start 更可靠。"""
    import os
    import platform
    import subprocess
    import time
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # 等服务真正可访问再打开，避免“连不上就关了”
    for _ in range(40):
        try:
            with opener.open(url, timeout=0.5) as resp:
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
    from werkzeug.serving import make_server

    if port:
        existing = find_existing_server(host, port)
        if existing:
            print(f"同一目录的平台已经运行：{existing}")
            if open_browser:
                _open_browser(existing)
            return
        port = _find_free_port(host, port)
    server = make_server(host, port, app, threaded=True)
    port = server.server_port
    url = f"http://{host}:{port}/"
    print("=" * 60)
    print(f"  {APP_NAME}")
    print(f"  本地地址: {url}")
    print("  请勿关闭本黑色窗口，关闭即停止服务。")
    print("=" * 60)
    if open_browser:
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()
    # 允许局域网访问可改为 host="0.0.0.0"；默认本机更安全
    try:
        server.serve_forever()
    finally:
        server.server_close()
