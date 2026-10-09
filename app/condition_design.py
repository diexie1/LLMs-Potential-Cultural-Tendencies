"""自然/受控运行条件的定义与可审计比较。

本模块不猜测供应商没有返回的实际默认值。它只描述请求中明确写入的
设置，并把“请求配置相同”和“平台最终生效值相同”严格区分开。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping

from .generation_config import GenerationConfig
from .prompting import (
    DEFAULT_PROMPT_CONTRACT,
    is_free_prompt_contract,
    normalize_prompt_contract,
    normalize_cultural_identity,
)


CONDITION_SCHEMA_VERSION = "condition-design-v2"
CONDITION_MODES = {"natural", "controlled", "custom"}

_CONTROL_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "top_k_mode",
    "min_p",
    "typical_p",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "n",
    "candidate_count",
    "best_of",
    "top_logprobs",
    "max_output_tokens",
    "stop",
    "thinking_mode",
    "reasoning_effort",
    "thinking_budget",
    "reasoning_summary",
    "response_format",
    "verbosity",
    "truncation",
    "service_tier",
    "prompt_cache_key",
    "store",
    "tools_enabled",
    "extra_body",
)


def normalize_condition_mode(value: Any) -> str:
    mode = str(value or "natural").strip().lower()
    if mode not in CONDITION_MODES:
        raise ValueError("condition_mode 必须是 natural、controlled 或 custom")
    return mode


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def condition_config_hash(config: GenerationConfig | Mapping[str, Any]) -> str:
    """Hash only the generation condition, excluding trial/run identifiers."""

    data = config.to_dict() if isinstance(config, GenerationConfig) else dict(config)
    return hashlib.sha256(_canonical(data).encode("utf-8")).hexdigest()


def _configured_fields(config: GenerationConfig) -> list[str]:
    data = config.to_dict()
    fields = []
    for name in _CONTROL_FIELDS:
        value = data.get(name)
        if name == "top_k_mode" and value == "provider_default":
            value = None
        if name == "top_k" and data.get("top_k_mode") == "provider_default":
            value = None
        if name == "thinking_mode" and value == "provider_default":
            value = None
        if name == "tools_enabled" and value is False:
            value = None
        if name == "extra_body" and not value:
            value = None
        if value is not None and value != [] and value != {}:
            fields.append(name)
    return fields


def build_condition_descriptor(
    mode: Any,
    config: GenerationConfig,
    prompt_contract: str = DEFAULT_PROMPT_CONTRACT,
    cultural_identity: str = "none",
) -> Dict[str, Any]:
    """Return the condition meaning stored in manifests and result records.

    The prompt condition is recorded because an optional instruction addendum
    changes the text sent to the model. No machine-enforced answer format is
    part of this condition.
    """

    normalized = normalize_condition_mode(mode)
    contract = normalize_prompt_contract(prompt_contract)
    identity = normalize_cultural_identity(cultural_identity)
    explicit = _configured_fields(config)
    if normalized == "natural":
        if explicit:
            interpretation = "natural_with_explicit_overrides"
            note = "标记为自然模式，但存在显式生成参数；不能解释为纯平台默认条件。"
        else:
            interpretation = "provider_default_requested"
            note = "采样与思考参数尽量省略，由供应商/模型决定；未返回的默认值仍可能未知。"
    elif normalized == "controlled":
        if explicit:
            interpretation = "controlled_explicit"
            note = "显式固定跨模型可比较的参数；未登记或不支持的字段应由严格校验阻止。"
        else:
            # A label alone cannot create a controlled condition. This can
            # happen when an operator selects the mode manually and clears
            # every generation field. Keep the run auditable, but make the
            # ambiguity visible instead of claiming that values were fixed.
            interpretation = "controlled_without_explicit_settings"
            note = "标记为受控模式，但没有任何显式生成参数；实际仍主要由平台/模型默认值决定。请应用受控预设或填写公共参数。"
    else:
        interpretation = "custom"
        note = "自定义条件；请以完整 generation_config 和请求快照为准。"
    if is_free_prompt_contract(contract):
        if contract == "free_scores_only":
            note += " 本实验条件在量表指导语部分增加一行只答分数提示；不附加机器格式或槽位要求。"
        else:
            note += " 按量表原文发送；不附加机器格式或槽位要求。"
    if identity != "none":
        country = "中国" if identity == "china" else "美国"
        note += f" 在原指导语前加入在{country}出生并生活的普通人身份提示。"
    return {
        "schema_version": CONDITION_SCHEMA_VERSION,
        "mode": normalized,
        "interpretation": interpretation,
        "note": note,
        "prompt_contract": contract,
        "cultural_identity": identity,
        "explicit_fields": explicit,
        "provider_default_fields": [name for name in _CONTROL_FIELDS if name not in explicit],
        "requested_config_sha256": condition_config_hash(config),
        "effective_value_status": "provider_default_or_unknown"
        if normalized == "natural"
        else "explicit_or_provider_rejected",
    }


def compare_condition_configs(
    left: GenerationConfig | Mapping[str, Any],
    right: GenerationConfig | Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare two requested configurations without claiming provider equivalence."""

    left_data = left.to_dict() if isinstance(left, GenerationConfig) else dict(left)
    right_data = right.to_dict() if isinstance(right, GenerationConfig) else dict(right)
    fields: Iterable[str] = sorted(set(left_data) | set(right_data))
    different = [name for name in fields if left_data.get(name) != right_data.get(name)]
    same = not different
    return {
        "schema_version": CONDITION_SCHEMA_VERSION,
        "requested_equivalent": same,
        "requested_difference_fields": different,
        "effective_equivalence": "unknown",
        "interpretation": (
            "两次请求配置完全相同；若 condition_mode 不同，只能视为标签重叠。"
            if same
            else "请求配置不同；平台最终是否改变仍需依据响应/目录元数据判断。"
        ),
        "left_config_sha256": condition_config_hash(left_data),
        "right_config_sha256": condition_config_hash(right_data),
    }
