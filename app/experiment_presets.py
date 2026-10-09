"""实验参数预设与参数说明。

预设只包含跨供应商可以明确解释的公共条件；厂商专属参数保持平台默认，
避免把某个平台的采样语义强行套到另一个平台。预设通过 Web API 暴露，
因此网页和后续打包版本使用同一份定义。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional

from .generation_config import GenerationConfig
from .prompting import (
    DEFAULT_PROMPT_CONTRACT,
    normalize_cultural_identity,
    normalize_prompt_contract,
)


PRESET_SCHEMA_VERSION = "experiment-preset-v2"
_PRESET_REQUIRED_FIELDS = {
    "id",
    "name",
    "description",
    "condition_mode",
    "prompt_config",
    "generation_config",
    "execution_config",
    "warnings",
}
_PRESET_CONDITION_MODES = {"natural", "controlled", "custom"}


def validate_preset_definition(preset_id: str, preset: Mapping[str, Any]) -> None:
    """Validate one registry entry before exposing it to the UI or runner.

    Presets are intentionally plain dictionaries so a future default can be
    added without changing the front end.  This boundary check keeps that
    extensibility safe: malformed entries fail with a useful error instead of
    appearing in the selector and failing only after a run has started.
    """

    if not isinstance(preset, Mapping):
        raise TypeError(f"实验预设 {preset_id!r} 必须是对象")
    missing = sorted(_PRESET_REQUIRED_FIELDS - set(preset))
    if missing:
        raise ValueError(
            f"实验预设 {preset_id!r} 缺少字段：{', '.join(missing)}"
        )
    if str(preset.get("id") or "").strip() != str(preset_id).strip():
        raise ValueError(f"实验预设 {preset_id!r} 的 id 与注册键不一致")
    for field in ("name", "description"):
        if not str(preset.get(field) or "").strip():
            raise ValueError(f"实验预设 {preset_id!r} 的 {field} 不能为空")
    condition_mode = str(preset.get("condition_mode") or "").strip().lower()
    if condition_mode not in _PRESET_CONDITION_MODES:
        raise ValueError(
            f"实验预设 {preset_id!r} 的 condition_mode 无效：{condition_mode!r}"
        )
    if not isinstance(preset.get("prompt_config"), Mapping):
        raise TypeError(f"实验预设 {preset_id!r} 的 prompt_config 必须是对象")
    normalize_prompt_config(preset.get("prompt_config"))
    if not isinstance(preset.get("generation_config"), Mapping):
        raise TypeError(f"实验预设 {preset_id!r} 的 generation_config 必须是对象")
    GenerationConfig.from_mapping(
        preset.get("generation_config"), fallback_temperature=1.0
    )
    if not isinstance(preset.get("execution_config"), Mapping):
        raise TypeError(f"实验预设 {preset_id!r} 的 execution_config 必须是对象")
    if not isinstance(preset.get("warnings"), (list, tuple)):
        raise TypeError(f"实验预设 {preset_id!r} 的 warnings 必须是列表")


# Only describe controls still used by the workbench.
PARAMETER_GUIDE: List[Dict[str, str]] = [
    {
        "id": "temperature",
        "label": "温度",
        "meaning": "控制回答的随机性；主分析默认0，温度条件默认1，可调整。",
        "comparison": "其他采样参数使用供应商默认值。"
    },
    {
        "id": "top_p",
        "label": "Top-p",
        "meaning": "默认1，可调整；实际可用范围由所选模型决定。",
        "comparison": "不静默忽略不支持的设置。"
    },
    {
        "id": "thinking_mode",
        "label": "思考模式",
        "meaning": "平台统一关闭思考模式。",
        "comparison": "所选模型没有已登记的关闭方式时，在请求前阻止运行。"
    },
    {
        "id": "concurrency",
        "label": "单量表 API 并发",
        "meaning": "同一个量表同时发送的试次请求数，范围1至500。",
        "comparison": "与量表并行数一起决定实际同时请求量。"
    },
    {
        "id": "scale_concurrency",
        "label": "量表并行数",
        "meaning": "同时运行的不同量表数，范围1至500。",
        "comparison": "同一量表的中英文依次运行，不增加该量表内的并发。"
    },
    {
        "id": "prompt_condition",
        "label": "指导语条件",
        "meaning": "主分析按原文发送；额外条件只增加只答分数提示或身份提示。",
        "comparison": "没有 JSON、槽位或机器解析格式要求。"
    }
]


_COMMON_GENERATION = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": None,
    "top_k_mode": "provider_default",
    "min_p": None,
    "typical_p": None,
    "seed": None,
    "presence_penalty": None,
    "frequency_penalty": None,
    "repetition_penalty": None,
    "logit_bias": None,
    "n": None,
    "candidate_count": None,
    "best_of": None,
    "top_logprobs": None,
    "max_output_tokens": None,
    "stop": None,
    "reasoning_effort": None,
    "thinking_budget": None,
    "reasoning_summary": None,
    "response_format": None,
    "verbosity": None,
    "truncation": None,
    "service_tier": None,
    "prompt_cache_key": None,
    "stream": False,
    "store": None,
    "tools_enabled": False,
    "extra_body": {},
}

# The platform sends the source scale wording as-is. The optional score-only
# experiment adds one instruction line; it is not an answer-format contract.
_FREE_PROMPT = {"contract": "unconstrained", "cultural_identity": "none"}
_FREE_SCORES_ONLY_PROMPT = {**_FREE_PROMPT, "contract": "free_scores_only"}

# The main analysis uses temperature = 0 and Top-p = 1; other sampling fields
# stay unset so each provider supplies its own defaults.
_FREE_GENERATION = {
    **_COMMON_GENERATION,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_output_tokens": None,
    "thinking_mode": "disabled",
}


_COMMON_EXECUTION = {
    "batch_en": 100,
    "batch_ch": 100,
    "concurrency": 5,
    "scale_concurrency": 2,
    "max_retries": 3,
    "max_group_attempts": 200,
    "sample_mode": "planned",
    "random_seed": 20260911,
    "strict_capabilities": False,
    "capture_network": True,
    "capture_model_catalog": True,
    "network_guard": "stop",
}


PRESETS: Dict[str, Dict[str, Any]] = {
    "free_response_v1": {
        "id": "free_response_v1",
        "name": "主分析",
        "description": (
            "按量表原文发送指导语、题目和情境材料，不额外规定回答格式或题数。"
            "temperature=0，Top-p=1；其他采样参数使用模型厂商默认值。"
        ),
        "condition_mode": "controlled",
        "prompt_config": dict(_FREE_PROMPT),
        "generation_config": dict(_FREE_GENERATION),
        "execution_config": dict(_COMMON_EXECUTION),
        "warnings": [
            "平台不增加回答格式或槽位限制；无法逐题拆分时仍保存完整原始回答。",
        ],
    },
    "free_response_temperature_1_v1": {
        "id": "free_response_temperature_1_v1",
        "name": "主分析·温度 1",
        "description": (
            "保持主分析设置，仅将 temperature 改为 1。"
        ),
        "condition_mode": "controlled",
        "prompt_config": dict(_FREE_PROMPT),
        "generation_config": {
            **_FREE_GENERATION,
            "temperature": 1.0,
        },
        "execution_config": dict(_COMMON_EXECUTION),
        "warnings": [
            "这是主分析的独立温度条件，不要与 temperature=0 的主分析结果混合计算。",
            "Top-p 保持 1，其他采样参数使用供应商默认值；不同供应商对 temperature=1 的实际效果可能不同。",
        ],
    },
    "free_response_scores_only_v1": {
        "id": "free_response_scores_only_v1",
        "name": "主分析·只答分数提示",
        "description": (
            "保持主分析的 temperature=0 和量表原文，仅在指导语部分增加一行"
            "“请仅给出题目要求的分数，不需要解释。”作为实验提示；"
            "不增加 JSON、编号或槽位要求。"
        ),
        "condition_mode": "controlled",
        "prompt_config": dict(_FREE_SCORES_ONLY_PROMPT),
        "generation_config": dict(_FREE_GENERATION),
        "execution_config": dict(_COMMON_EXECUTION),
        "warnings": [
            "此条件只多一行指导语，不要求模型按固定格式作答；原始回答始终完整保留。",
        ],
    },
    "china_identity_v1": {
        "id": "china_identity_v1",
        "name": "中国身份条件",
        "description": "在原量表指导语前加入在中国出生并生活的普通人身份提示；中英文自动对应，其他设置沿用主分析。",
        "condition_mode": "controlled",
        "prompt_config": {**_FREE_PROMPT, "cultural_identity": "china"},
        "generation_config": dict(_FREE_GENERATION),
        "execution_config": dict(_COMMON_EXECUTION),
        "warnings": [],
    },
    "usa_identity_v1": {
        "id": "usa_identity_v1",
        "name": "美国身份条件",
        "description": "在原量表指导语前加入在美国出生并生活的普通人身份提示；中英文自动对应，其他设置沿用主分析。",
        "condition_mode": "controlled",
        "prompt_config": {**_FREE_PROMPT, "cultural_identity": "usa"},
        "generation_config": dict(_FREE_GENERATION),
        "execution_config": dict(_COMMON_EXECUTION),
        "warnings": [],
    },
}


def list_presets() -> List[Dict[str, Any]]:
    """Return JSON-safe copies for the Web UI."""

    for preset_id, preset in PRESETS.items():
        validate_preset_definition(preset_id, preset)
    return [deepcopy(item) for item in PRESETS.values()]


def get_preset(preset_id: str) -> Optional[Dict[str, Any]]:
    item = PRESETS.get(str(preset_id or "").strip())
    if item is not None:
        validate_preset_definition(str(preset_id or "").strip(), item)
    return deepcopy(item) if item is not None else None


def apply_preset(preset_id: str) -> Dict[str, Any]:
    """Validate and return a complete preset without mutating user config."""

    preset = get_preset(preset_id)
    if preset is None:
        raise KeyError(f"未知实验预设：{preset_id}")
    # Validation catches accidental drift in the preset before it reaches the
    # request builder. The UI still receives the full explicit/omitted map.
    preset["generation_config"] = GenerationConfig.from_mapping(
        preset.get("generation_config"), fallback_temperature=1.0
    ).to_dict()
    preset["condition_mode"] = str(preset.get("condition_mode") or "custom")
    preset["prompt_config"] = normalize_prompt_config(preset.get("prompt_config"))
    return preset


def normalize_prompt_config(value: Any) -> Dict[str, Any]:
    """Return a prompt condition without machine-enforced answer contracts."""

    if not isinstance(value, Mapping):
        value = {}
    try:
        contract = normalize_prompt_contract(value.get("contract"))
    except ValueError:
        contract = DEFAULT_PROMPT_CONTRACT
    if contract == "strict":
        contract = "unconstrained"
    return {
        "contract": contract,
        "cultural_identity": normalize_cultural_identity(value.get("cultural_identity")),
    }


def parameter_guide() -> List[Dict[str, str]]:
    return deepcopy(PARAMETER_GUIDE)
