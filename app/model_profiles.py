"""模型文档、参数解析状态和可审计的模型说明。

这个模块不把平台文档中的默认值冒充成一次请求的真实生效值。每个参数
都同时保留：本次配置、实际发送的值、API 返回的可观察值以及官方文档值。
这样即使供应商不在响应中回填默认参数，结果仍能清楚区分“已知”和“未知”。
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from .generation_config import GenerationConfig
from .provenance import canonical_hash


MODEL_PROFILE_SCHEMA_VERSION = "model-profile-v1"
OFFICIAL_DOC_CATALOG_VERSION = "official-doc-catalog-v1"
DOC_CATALOG_AS_OF = "2026-09-11"


# These are the settings that affect a generation request in this application.
# ``stream`` is retained because it is part of the actual request even though
# the formal batch runner always sends false.
PARAMETER_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "logit_bias",
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
    "stream",
    "store",
    "tools_enabled",
    "extra_body",
)

PARAMETER_LABELS = {
    "temperature": "Temperature",
    "top_p": "Top-p",
    "top_k": "Top-k",
    "min_p": "Min-p",
    "typical_p": "Typical-p",
    "seed": "Seed",
    "presence_penalty": "Presence penalty",
    "frequency_penalty": "Frequency penalty",
    "repetition_penalty": "Repetition penalty",
    "logit_bias": "Logit bias",
    "n": "候选数 n",
    "candidate_count": "Candidate count",
    "best_of": "Best of",
    "top_logprobs": "Top logprobs",
    "max_output_tokens": "最大输出 tokens",
    "stop": "Stop",
    "thinking_mode": "思考模式",
    "reasoning_effort": "推理档位",
    "thinking_budget": "思考预算",
    "reasoning_summary": "推理摘要",
    "response_format": "响应格式",
    "verbosity": "Verbosity",
    "truncation": "Truncation",
    "service_tier": "Service tier",
    "prompt_cache_key": "Prompt cache key",
    "stream": "Stream",
    "store": "Store",
    "tools_enabled": "Tools",
    "extra_body": "厂商扩展参数",
}


# The URLs are official documentation pages. Values in ``documented_defaults``
# are deliberately limited to values explicitly stated there. ``None`` means
# the documentation says the value is model-dependent or does not expose it.
OFFICIAL_DOCUMENTATION: Dict[str, Dict[str, Any]] = {
    "deepseek": {
        "provider_name": "DeepSeek",
        "documents": [
            {
                "title": "Chat Completions API",
                "url": "https://api-docs.deepseek.com/api/create-chat-completion/",
                "scope": "OpenAI-compatible chat/completions",
            },
            {
                "title": "Thinking Mode",
                "url": "https://api-docs.deepseek.com/guides/thinking_mode/",
                "scope": "thinking-mode compatibility",
            },
        ],
        "documented_defaults": {"temperature": 1, "top_p": 1},
        "notes": [
            "官方 Chat Completions 文档将 temperature 和 top_p 的默认值列为 1。",
            "官方文档说明关闭思考后 Top-p 固定为 1；因此平台不发送该字段，并阻止非 1 的设置，避免参数被忽略。",
            "官方文档说明思考模式下 temperature、presence_penalty、frequency_penalty 无效；平台关闭思考后仍可发送 temperature。",
        ],
    },
    "openai": {
        "provider_name": "OpenAI",
        "documents": [
            {
                "title": "Chat API reference",
                "url": "https://developers.openai.com/api/reference/resources/chat",
                "scope": "Chat Completions request parameters",
            },
            {
                "title": "Models API reference",
                "url": "https://platform.openai.com/docs/api-reference/models/object",
                "scope": "model identity metadata",
            },
        ],
        "documented_defaults": {"temperature": 1, "top_p": 1, "n": 1},
        "notes": [
            "模型列表/模型对象接口主要提供 id、created、owned_by 等身份字段，不保证回填一次请求最终采用的采样默认值。",
            "推理模型可能有独立的 reasoning 参数规则；不能用普通聊天模型默认值替代。",
        ],
    },
    "qwen": {
        "provider_name": "通义千问（百炼）",
        "documents": [
            {
                "title": "Qwen OpenAI-compatible Chat Completions",
                "url": "https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions",
                "scope": "OpenAI-compatible chat/completions",
            },
            {
                "title": "Qwen API reference",
                "url": "https://help.aliyun.com/zh/model-studio/qwen-api-reference/",
                "scope": "model and parameter overview",
            },
        ],
        # These two values are documented for the qwen-plus series in the
        # non-thinking family. The resolver labels them as family/conditional,
        # never as an exact immutable model fact.
        "family_defaults": {
            "qwen-plus": {"top_p": 0.8, "top_k": 20},
            "qwen-turbo": {"top_p": 0.8, "top_k": 20},
            "qwen-max": {"top_p": 0.8, "top_k": 20},
        },
        "documented_defaults": {},
        "notes": [
            "百炼文档按模型系列和思考模式列出默认值；省略参数时也可能继承控制台/部署配置。",
            "temperature 与 top_p 官方建议只调整一个；本项目仍会逐项记录实际发送值。",
        ],
    },
    "gemini": {
        "provider_name": "Google Gemini",
        "documents": [
            {
                "title": "GenerateContent generation config",
                "url": "https://ai.google.dev/api/generate-content",
                "scope": "temperature/topP/topK/candidateCount/seed",
            },
            {
                "title": "Gemini model resource",
                "url": "https://ai.google.dev/api/models",
                "scope": "model-specific default metadata",
            },
        ],
        "documented_defaults": {"candidate_count": 1},
        "notes": [
            "官方文档说明 temperature、topP、topK 和最大输出长度的默认值随具体 Model 变化，应从 Model 元数据读取。",
            "若当前兼容端点没有返回 Model 元数据，本次记录会明确标为未观测，而不是猜测数值。",
        ],
    },
}


def _json_copy(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception:
        return value


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_family_defaults(provider: str, model: str) -> Dict[str, Any]:
    docs = OFFICIAL_DOCUMENTATION.get(provider, {})
    families = docs.get("family_defaults") or {}
    lowered = model.casefold()
    # Longest prefix wins, which keeps a dated snapshot such as qwen-plus-*
    # under the same documented family while remaining conservative.
    matches = [name for name in families if lowered.startswith(name.casefold())]
    if not matches:
        return {}
    family = max(matches, key=len)
    return dict(families[family])


def _extract_observed_parameters(response_snapshot: Any) -> Dict[str, Any]:
    """Read only explicit provider fields that can represent resolved config.

    Most OpenAI-compatible responses do not expose this information. We do not
    inspect arbitrary nested ``parameters`` objects because they may describe a
    tool schema rather than generation settings.
    """

    if not isinstance(response_snapshot, Mapping):
        return {}
    candidates = []
    for key in (
        "resolved_generation_config",
        "generation_config",
        "generationConfig",
        "effective_generation_config",
        "effectiveGenerationConfig",
    ):
        value = response_snapshot.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    if not candidates:
        return {}
    aliases = {
        "topP": "top_p",
        "topK": "top_k",
        "candidateCount": "candidate_count",
        "maxOutputTokens": "max_output_tokens",
        "presencePenalty": "presence_penalty",
        "frequencyPenalty": "frequency_penalty",
        "repetitionPenalty": "repetition_penalty",
        "thinkingBudget": "thinking_budget",
        "enable_thinking": "thinking_mode",
    }
    observed: Dict[str, Any] = {}
    for source in candidates:
        for key, value in source.items():
            name = aliases.get(str(key), str(key))
            if name in PARAMETER_FIELDS:
                if name == "thinking_mode" and isinstance(value, bool):
                    value = "enabled" if value else "disabled"
                observed[name] = _json_copy(value)
    return observed


def _configured_values(config: GenerationConfig) -> Dict[str, Any]:
    values = config.to_dict()
    # A mode is itself the user's explicit choice even when its value is null.
    values["top_k"] = (
        config.top_k
        if config.top_k_mode == "explicit"
        else (None if config.top_k_mode == "provider_default" else "disabled")
    )
    values["thinking_mode"] = config.thinking_mode
    values["store"] = config.store
    return values


def _sent_value(field: str, request_snapshot: Mapping[str, Any]) -> Any:
    if field == "top_k":
        if "top_k" in request_snapshot:
            return request_snapshot["top_k"]
        extra = request_snapshot.get("extra_body")
        if isinstance(extra, Mapping) and "top_k" in extra:
            return extra["top_k"]
        return None
    if field == "thinking_mode":
        extra = request_snapshot.get("extra_body")
        if isinstance(extra, Mapping):
            thinking = extra.get("thinking")
            if isinstance(thinking, Mapping) and "type" in thinking:
                return str(thinking["type"])
            if "enable_thinking" in extra:
                return "enabled" if bool(extra["enable_thinking"]) else "disabled"
        return None
    if field in request_snapshot:
        return request_snapshot[field]
    # Non-standard settings are passed through extra_body by the builder.
    extra = request_snapshot.get("extra_body")
    if isinstance(extra, Mapping) and field in extra:
        return extra[field]
    return None


def _value_text(value: Any) -> str:
    if value is None:
        return "未设置"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _parameter_resolution(
    *,
    provider: str,
    model: str,
    field: str,
    configured: Any,
    request_snapshot: Mapping[str, Any],
    observed: Mapping[str, Any],
    exact_documented_defaults: Mapping[str, Any],
    family_defaults: Mapping[str, Any],
) -> Dict[str, Any]:
    sent = _sent_value(field, request_snapshot)
    extra = request_snapshot.get("extra_body")
    sent_present = sent is not None or (
        field == "store" and "store" in request_snapshot
    ) or (field == "stream" and "stream" in request_snapshot) or (
        field == "top_k"
        and isinstance(extra, Mapping)
        and "top_k" in extra
    )
    observed_present = field in observed
    if observed_present:
        resolution = "observed_from_response"
        effective = observed[field]
        source = "provider_response"
    elif sent_present:
        resolution = "explicit_request"
        effective = sent
        source = "request_snapshot"
    elif configured == "disabled" or (
        configured is False
        and field in {"thinking_mode", "store", "top_k", "tools_enabled"}
    ):
        resolution = "explicitly_disabled"
        effective = configured
        source = "generation_config"
    elif configured == "provider_default" or configured is None or configured == {}:
        if field in exact_documented_defaults:
            resolution = "documented_default_not_observed"
            effective = exact_documented_defaults[field]
            source = "official_documentation"
        elif field in family_defaults:
            resolution = "family_default_not_observed"
            effective = family_defaults[field]
            source = "official_documentation_family"
        else:
            resolution = "provider_default_not_returned"
            effective = None
            source = "not_exposed"
    else:
        resolution = "configured_but_not_sent"
        effective = configured
        source = "generation_config"
    item = {
        "label": PARAMETER_LABELS.get(field, field),
        "configured_value": _json_copy(configured),
        "sent": bool(sent_present),
        "sent_value": _json_copy(sent),
        "observed_value": _json_copy(observed.get(field)) if observed_present else None,
        "documented_default": _json_copy(exact_documented_defaults.get(field)),
        "documented_family_default": _json_copy(family_defaults.get(field)),
        "effective_value": _json_copy(effective),
        "resolution": resolution,
        "source": source,
    }
    if field == "tools_enabled" and configured is False and not sent_present:
        item["note"] = "当前量表运行器不发送工具定义；此字段仅表示工具保持关闭。"
    if field in {"temperature", "top_p"} and provider == "deepseek":
        item["note"] = (
            "关闭思考后 Top-p 固定为 1；本请求省略该字段。"
            if field == "top_p"
            else "思考模式下该参数无效；本平台已关闭思考模式。"
        )
    return item


def build_model_profile(
    provider: str,
    requested_model: str,
    *,
    config: GenerationConfig,
    request_snapshot: Optional[Mapping[str, Any]] = None,
    response_snapshot: Any = None,
    returned_model: Optional[str] = None,
    model_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a complete, JSON-safe model/settings profile for one request."""

    provider = str(provider or "").strip()
    requested_model = str(requested_model or "").strip()
    request = dict(request_snapshot or {})
    docs = OFFICIAL_DOCUMENTATION.get(provider)
    if docs is None:
        docs = {
            "provider_name": provider or "未知平台",
            "documents": [],
            "documented_defaults": {},
            "notes": ["尚未登记该平台的官方参数文档。"],
        }
    exact_defaults = dict(docs.get("documented_defaults") or {})
    family_defaults = _model_family_defaults(provider, requested_model)
    observed = _extract_observed_parameters(response_snapshot)
    configured = _configured_values(config)
    parameters = {
        field: _parameter_resolution(
            provider=provider,
            model=requested_model,
            field=field,
            configured=configured.get(field),
            request_snapshot=request,
            observed=observed,
            exact_documented_defaults=exact_defaults,
            family_defaults=family_defaults,
        )
        for field in PARAMETER_FIELDS
    }
    metadata = dict(model_metadata or {})
    resolved_count = sum(
        item["resolution"] in {"observed_from_response", "explicit_request", "explicitly_disabled"}
        for item in parameters.values()
    )
    unknown_defaults = sum(
        item["resolution"] == "provider_default_not_returned"
        for item in parameters.values()
    )
    returned = returned_model or metadata.get("returned_model")
    identity_status = metadata.get("model_identity_status")
    if not identity_status:
        if metadata.get("is_pinned") is True and metadata.get("snapshot_id"):
            identity_status = "pinned_snapshot_verified"
        elif returned:
            # A returned model string is valuable evidence, but it is not by
            # itself proof that a rolling alias resolved to an immutable build.
            identity_status = "alias_or_snapshot_unverified"
        else:
            identity_status = "model_identity_not_returned"
    identity = {
        "provider": provider,
        "requested_model": requested_model,
        "returned_model": returned,
        "identity_status": identity_status,
        "snapshot_id": metadata.get("snapshot_id"),
        "snapshot_source": metadata.get("snapshot_source"),
        "is_pinned": metadata.get("is_pinned"),
        "deployment_id": metadata.get("deployment_id"),
        "provider_region": metadata.get("provider_region"),
        "endpoint": metadata.get("endpoint"),
        "model_catalog_snapshot_id": metadata.get("model_catalog_snapshot_id"),
        "account_snapshot_id": metadata.get("account_snapshot_id"),
        "condition_mode": metadata.get("condition_mode"),
        "condition_config_sha256": metadata.get("condition_config_sha256"),
        "response_id": metadata.get("response_id"),
        "client_request_id": metadata.get("client_request_id"),
        "provider_request_id": metadata.get("provider_request_id"),
        "attempt_id": metadata.get("attempt_id"),
        "network_snapshot_id": metadata.get("network_snapshot_id"),
        "system_fingerprint": metadata.get("system_fingerprint"),
        "response_headers": metadata.get("response_headers"),
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    provider_name = str(docs.get("provider_name") or provider)
    visible = [
        f"{item['label']}={_value_text(item['effective_value'])}"
        for item in parameters.values()
        if item["sent"]
        or item["configured_value"] not in (None, "provider_default", {})
    ]
    unresolved_labels = [
        item["label"]
        for item in parameters.values()
        if item["resolution"] == "provider_default_not_returned"
    ]
    documented_reference = []
    for field, item in parameters.items():
        reference = item.get("documented_default")
        reference_scope = "官方文档"
        if reference is None:
            reference = item.get("documented_family_default")
            reference_scope = "官方文档模型系列"
        if reference is not None:
            documented_reference.append(
                f"{item['label']}={_value_text(reference)}（{reference_scope}，非本次响应实测）"
            )
    description = (
        f"{provider_name} / {requested_model}；"
        f"API返回模型={identity.get('returned_model', '未返回')}；"
        f"模型身份状态={identity_status}；"
        f"本次实际发送设置：{'；'.join(visible) if visible else '无显式生成参数'}。"
    )
    if documented_reference:
        description += " 官方文档默认参考：" + "；".join(documented_reference) + "。"
    if unresolved_labels:
        description += (
            " 未能从本次 API 响应取得平台默认实际值的参数："
            + "、".join(unresolved_labels)
            + "；这些参数只能标记为平台默认，不能声称已确定具体数值。"
        )
    return {
        "schema_version": MODEL_PROFILE_SCHEMA_VERSION,
        "catalog_version": OFFICIAL_DOC_CATALOG_VERSION,
        "catalog_as_of": DOC_CATALOG_AS_OF,
        "profile_created_at_utc": _now_utc(),
        "identity": identity,
        "official_documentation": {
            "provider": provider,
            "documents": _json_copy(docs.get("documents") or []),
            "extraction_method": "curated_from_official_documentation",
            "extracted_at": DOC_CATALOG_AS_OF,
            "catalog_hash": canonical_hash(docs),
            "notes": _json_copy(docs.get("notes") or []),
            "model_scope": "provider_or_family",
            "model_match": "family" if family_defaults else "provider",
        },
        "parameters": parameters,
        "summary": {
            "description": description,
            "resolved_or_explicit_count": resolved_count,
            "provider_default_not_returned_count": unknown_defaults,
            "response_resolved_parameters": sorted(observed),
            "actual_request_parameter_source": "request_snapshot",
            "documented_default_reference": documented_reference,
            "identity_status": identity_status,
        },
    }


def model_profile_key(provider: str, model: str) -> str:
    return f"{str(provider or '').strip()}/{str(model or '').strip()}"
