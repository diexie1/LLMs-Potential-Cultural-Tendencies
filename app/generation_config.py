"""模型生成配置、能力规则和请求参数解析。

该模块只描述实验意图和已登记的兼容性，不执行网络请求。厂商适配器负责把
``GenerationConfig`` 转成真实的 OpenAI-compatible 请求字段。
"""

from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


PARAMETER_MODES = {"provider_default", "explicit", "disabled"}
THINKING_MODES = {"provider_default", "enabled", "disabled"}


@dataclass(frozen=True)
class GenerationConfig:
    """跨厂商的生成设置。

    ``None`` means the parameter is not explicitly configured. ``top_k_mode``
    and ``thinking_mode`` preserve the important distinction between omitted
    and explicitly disabled values.
    """

    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    top_k_mode: str = "provider_default"
    min_p: Optional[float] = None
    typical_p: Optional[float] = None
    seed: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, int]] = None
    # Omitted means the provider's single-candidate default; explicit 1 can be
    # selected for a formal condition when the capability registry confirms it.
    n: Optional[int] = None
    candidate_count: Optional[int] = None
    best_of: Optional[int] = None
    top_logprobs: Optional[int] = None
    max_output_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    thinking_mode: str = "provider_default"
    reasoning_effort: Optional[str] = None
    thinking_budget: Optional[int] = None
    reasoning_summary: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    verbosity: Optional[str] = None
    truncation: Optional[str] = None
    service_tier: Optional[str] = None
    prompt_cache_key: Optional[str] = None
    stream: bool = False
    store: Optional[bool] = None
    tools_enabled: bool = False
    extra_body: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.top_k_mode not in PARAMETER_MODES:
            raise ValueError(f"top_k_mode 必须是 {sorted(PARAMETER_MODES)}")
        if self.thinking_mode not in THINKING_MODES:
            raise ValueError(f"thinking_mode 必须是 {sorted(THINKING_MODES)}")
        if self.temperature is not None and not 0 <= float(self.temperature) <= 2:
            raise ValueError("temperature 必须在 0 到 2 之间")
        if self.top_p is not None and not 0 <= float(self.top_p) <= 1:
            raise ValueError("top_p 必须在 0 到 1 之间")
        for name, value in (("min_p", self.min_p), ("typical_p", self.typical_p)):
            if value is not None and not 0 <= float(value) <= 1:
                raise ValueError(f"{name} 必须在 0 到 1 之间")
        if self.top_k_mode == "explicit" and (
            self.top_k is None or int(self.top_k) < 1
        ):
            raise ValueError("top_k_mode=explicit 时 top_k 必须是正整数")
        if self.seed is not None and int(self.seed) < 0:
            raise ValueError("seed 不能为负数")
        if self.n is not None and int(self.n) < 1:
            raise ValueError("n 必须至少为 1")
        if self.candidate_count is not None and int(self.candidate_count) < 1:
            raise ValueError("candidate_count 必须至少为 1")
        if self.best_of is not None and int(self.best_of) < 1:
            raise ValueError("best_of 必须至少为 1")
        if self.top_logprobs is not None and not 0 <= int(self.top_logprobs) <= 20:
            raise ValueError("top_logprobs 必须在 0 到 20 之间")
        if self.max_output_tokens is not None and int(self.max_output_tokens) < 1:
            raise ValueError("max_output_tokens 必须为正整数")
        if self.thinking_budget is not None and int(self.thinking_budget) < 1:
            raise ValueError("thinking_budget 必须为正整数")
        if self.reasoning_effort and self.thinking_budget is not None:
            raise ValueError("reasoning_effort 与 thinking_budget 不能同时设置")
        if not isinstance(self.extra_body, ABCMapping):
            raise ValueError("extra_body 必须是 JSON 对象")
        for name, value in (("logit_bias", self.logit_bias), ("response_format", self.response_format)):
            if value is not None and not isinstance(value, ABCMapping):
                raise ValueError(f"{name} 必须是 JSON 对象")

    @classmethod
    def from_mapping(
        cls, value: Optional[Mapping[str, Any]], *, fallback_temperature: float = 0.7
    ) -> "GenerationConfig":
        """从用户配置读取，保留0、False和空列表等合法显式值。"""

        data = dict(value or {})
        kwargs: Dict[str, Any] = {}
        fields = {
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
        }
        for name in fields:
            if name in data:
                kwargs[name] = data[name]
        if "temperature" not in kwargs:
            kwargs["temperature"] = fallback_temperature
        # A numeric top_k from an older/simple config means an explicit value;
        # an omitted field still means provider default.
        if "top_k" in kwargs and "top_k_mode" not in kwargs:
            if kwargs["top_k"] is not None:
                kwargs["top_k_mode"] = "explicit"
        if kwargs.get("stop") is not None:
            kwargs["stop"] = [str(item) for item in kwargs["stop"]]
        if kwargs.get("extra_body") is None:
            kwargs["extra_body"] = {}
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_temperature_fallback(self, value: float) -> "GenerationConfig":
        """Compatibility bridge for the legacy single-temperature UI field."""

        if self.temperature is None:
            return replace(self, temperature=value)
        return self


def simple_platform_generation_config(
    value: Optional[Mapping[str, Any]], *, fallback_temperature: float = 0.7
) -> GenerationConfig:
    """Keep the user-facing controls and reset all other sampling fields.

    The platform exposes temperature and top-p only. A missing top-p means 1.0;
    every other optional generation field follows the provider default. Thinking
    is handled separately by the platform's verified provider adapter.
    """

    if isinstance(value, GenerationConfig):
        value = value.to_dict()
    if value is not None and not isinstance(value, ABCMapping):
        raise ValueError("生成参数必须是 JSON 对象")
    data = value or {}
    temperature = data.get("temperature", fallback_temperature)
    top_p = data.get("top_p")
    return GenerationConfig(
        temperature=None if temperature is None else float(temperature),
        top_p=1.0 if top_p is None else float(top_p),
        thinking_mode="disabled",
        stream=data.get("stream") is True,
    )


def force_thinking_off(config: GenerationConfig) -> GenerationConfig:
    """Return the platform's fixed no-thinking generation configuration."""

    extra = dict(config.extra_body or {})
    for key in ("thinking", "enable_thinking", "thinking_budget", "reasoning"):
        extra.pop(key, None)
    google = extra.get("google")
    if isinstance(google, ABCMapping):
        google = dict(google)
        google.pop("thinking_config", None)
        google.pop("thinkingConfig", None)
        if google:
            extra["google"] = google
        else:
            extra.pop("google", None)
    return replace(
        config,
        thinking_mode="disabled",
        reasoning_effort=None,
        thinking_budget=None,
        reasoning_summary=None,
        extra_body=extra,
    )


@dataclass(frozen=True)
class ParameterRule:
    status: str = "unknown"  # supported/unsupported/ignored/conditional/unknown
    wire_field: Optional[str] = None
    note: str = ""
    allowed_modes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityProfile:
    provider: str
    model: str
    parameters: Dict[str, ParameterRule]
    source: str = "local-registry"
    as_of: str = "2026-09-11"


def _common_profile(provider: str, model: str) -> CapabilityProfile:
    return CapabilityProfile(
        provider=provider,
        model=model,
        parameters={
            "temperature": ParameterRule("supported", "temperature"),
            "top_p": ParameterRule("supported", "top_p"),
            "top_k": ParameterRule("unknown"),
            "min_p": ParameterRule("unknown"),
            "typical_p": ParameterRule("unknown"),
            "seed": ParameterRule("unknown", "seed"),
            "presence_penalty": ParameterRule("unknown", "presence_penalty"),
            "frequency_penalty": ParameterRule("unknown", "frequency_penalty"),
            "repetition_penalty": ParameterRule("unknown"),
            "logit_bias": ParameterRule("unknown", "logit_bias"),
            "n": ParameterRule("supported", "n"),
            "candidate_count": ParameterRule("unknown"),
            "best_of": ParameterRule("unknown"),
            "top_logprobs": ParameterRule("unknown", "top_logprobs"),
            "max_output_tokens": ParameterRule("supported"),
            "stop": ParameterRule("unknown", "stop"),
            "thinking_mode": ParameterRule("unknown"),
            "reasoning_effort": ParameterRule("unknown"),
            "thinking_budget": ParameterRule("unknown"),
            "reasoning_summary": ParameterRule("unknown"),
            "response_format": ParameterRule("unknown", "response_format"),
            "verbosity": ParameterRule("unknown"),
            "truncation": ParameterRule("unknown", "truncation"),
            "service_tier": ParameterRule("unknown", "service_tier"),
            "prompt_cache_key": ParameterRule("unknown", "prompt_cache_key"),
            "store": ParameterRule("unknown", "store"),
        },
    )


def _profile(provider: str, model: str, **overrides: ParameterRule) -> CapabilityProfile:
    base = _common_profile(provider, model)
    params = dict(base.parameters)
    params.update(overrides)
    return replace(base, parameters=params)


# This is deliberately conservative. Exact new model IDs must be added after
# a documentation check or a paid-free preflight; names are never used as
# proof of capability in strict mode.
CAPABILITY_REGISTRY: Dict[Tuple[str, str], CapabilityProfile] = {
    ("deepseek", "deepseek-flash"): _profile(
        "deepseek",
        "deepseek-flash",
        temperature=ParameterRule("conditional", "temperature", "thinking模式下无效"),
        top_p=ParameterRule("conditional", "top_p", "关闭thinking后固定为1"),
        presence_penalty=ParameterRule("ignored", note="thinking模式下无效"),
        frequency_penalty=ParameterRule("ignored", note="thinking模式下无效"),
        thinking_mode=ParameterRule("supported"),
        reasoning_effort=ParameterRule("supported"),
    ),
    ("deepseek", "deepseek-v4-pro"): _profile(
        "deepseek",
        "deepseek-v4-pro",
        temperature=ParameterRule("conditional", "temperature", "thinking模式下无效；关闭thinking后可用"),
        top_p=ParameterRule("conditional", "top_p", "关闭thinking后固定为1"),
        thinking_mode=ParameterRule("supported"),
        reasoning_effort=ParameterRule("supported"),
    ),
    ("openai", "gpt-4o"): _profile("openai", "gpt-4o", seed=ParameterRule("supported", "seed")),
    ("openai", "gpt-4o-mini"): _profile("openai", "gpt-4o-mini", seed=ParameterRule("supported", "seed")),
    ("openai", "gpt-4.1"): _profile("openai", "gpt-4.1", seed=ParameterRule("supported", "seed")),
    ("openai", "o3-mini"): _profile(
        "openai",
        "o3-mini",
        temperature=ParameterRule("unsupported", note="推理模型不接受该字段"),
        top_p=ParameterRule("unsupported", note="推理模型不接受该字段"),
        thinking_mode=ParameterRule("conditional", note="由推理档位控制"),
        reasoning_effort=ParameterRule("supported"),
    ),
    ("openai", "gpt-5.4"): _profile(
        "openai",
        "gpt-5.4",
        thinking_mode=ParameterRule("supported"),
        reasoning_effort=ParameterRule("supported", allowed_modes=("none",)),
    ),
    ("openai", "gpt-5.5"): _profile(
        "openai",
        "gpt-5.5",
        thinking_mode=ParameterRule("supported"),
        reasoning_effort=ParameterRule("supported", allowed_modes=("none",)),
    ),
    ("qwen", "qwen-plus"): _profile("qwen", "qwen-plus", top_k=ParameterRule("supported", "top_k"), thinking_mode=ParameterRule("conditional")),
    ("qwen", "qwen-turbo"): _profile("qwen", "qwen-turbo", top_k=ParameterRule("supported", "top_k")),
    ("qwen", "qwen-max"): _profile("qwen", "qwen-max", top_k=ParameterRule("supported", "top_k")),
    # Verified against Alibaba Cloud Model Studio's qwen3.7-flash and
    # DashScope API reference on 2026-09-13. Keep this exact ID separate from
    # dated snapshots and other Qwen models; they require their own review.
    ("qwen", "qwen3.7-flash"): replace(
        _profile(
            "qwen",
            "qwen3.7-flash",
            temperature=ParameterRule(
                "conditional",
                "temperature",
                "支持设置；平台默认值随思考模式变化",
            ),
            top_p=ParameterRule(
                "conditional",
                "top_p",
                "支持设置；平台默认值随思考模式变化",
            ),
            top_k=ParameterRule("supported", "top_k"),
            seed=ParameterRule("supported", "seed"),
            presence_penalty=ParameterRule("supported", "presence_penalty"),
            n=ParameterRule(
                "unsupported",
                note="百炼当前 n 参数支持列表不包含 Qwen3.7-Flash",
            ),
            stop=ParameterRule("supported", "stop"),
            response_format=ParameterRule("supported", "response_format"),
            thinking_mode=ParameterRule("supported", "enable_thinking"),
            thinking_budget=ParameterRule("conditional", "thinking_budget"),
            reasoning_effort=ParameterRule(
                "unsupported",
                note="reasoning_effort 的官方支持说明针对 Qwen3.8；Qwen3.7 使用 enable_thinking/thinking_budget",
            ),
        ),
        source="Alibaba Cloud Model Studio official documentation",
        as_of="2026-09-13",
    ),
    ("qwen", "qwen3.8-flash"): _profile(
        "qwen",
        "qwen3.8-flash",
        thinking_mode=ParameterRule("supported"),
        reasoning_effort=ParameterRule("supported", allowed_modes=("none",)),
    ),
    ("gemini", "gemini-2.0-flash"): _profile("gemini", "gemini-2.0-flash", top_k=ParameterRule("supported", "top_k")),
    ("gemini", "gemini-2.5-flash"): _profile("gemini", "gemini-2.5-flash", top_k=ParameterRule("supported", "top_k"), thinking_mode=ParameterRule("conditional"), reasoning_effort=ParameterRule("supported", allowed_modes=("none",))),
    ("gemini", "gemini-2.5-pro"): _profile("gemini", "gemini-2.5-pro", top_k=ParameterRule("supported", "top_k"), thinking_mode=ParameterRule("conditional")),
}


def thinking_off_strategy(provider: str, model: str) -> Optional[str]:
    """Return a verified API strategy for disabling a model's reasoning mode."""

    provider = str(provider or "").strip().casefold()
    model = str(model or "").strip().casefold()
    if not model:
        return None
    if provider == "deepseek":
        if model.startswith(("deepseek-v4-pro", "deepseek-v4-flash", "deepseek-flash")):
            return "deepseek_toggle"
        return None
    if provider == "qwen":
        if model.startswith("qwen3.8"):
            return "reasoning_effort_none"
        if model.startswith(("qwen3.7", "qwen3.6", "qwen3.5", "qwen3-", "qwen3_")):
            return "qwen_toggle"
        if model in {"qwen-plus", "qwen-turbo", "qwen-max"}:
            return "intrinsic_off"
        return None
    if provider == "openai":
        if model.startswith(("gpt-4o", "gpt-4.1")):
            return "intrinsic_off"
        if model == "gpt-5.4" or model.startswith("gpt-5.4-"):
            return "reasoning_effort_none"
        if model == "gpt-5.5" or model.startswith("gpt-5.5-"):
            return "reasoning_effort_none"
        return None
    if provider == "gemini":
        if model == "gemini-2.0-flash" or model.startswith("gemini-2.0-flash-"):
            return "intrinsic_off"
        if model == "gemini-2.5-flash" or model.startswith("gemini-2.5-flash-"):
            return "reasoning_effort_none"
        return None
    return None


def get_capability(provider: str, model: str) -> Optional[CapabilityProfile]:
    return CAPABILITY_REGISTRY.get((str(provider).strip(), str(model).strip()))


def validate_generation_config(
    provider: str,
    model: str,
    config: GenerationConfig,
    *,
    strict: bool = False,
) -> List[str]:
    """Return errors; unknown capabilities are errors only in strict mode."""

    errors: List[str] = []
    profile = get_capability(provider, model)
    if (
        str(provider).strip().casefold() == "deepseek"
        and config.thinking_mode == "disabled"
        and config.top_p is not None
        and float(config.top_p) != 1.0
    ):
        errors.append(
            f"{provider}/{model} 关闭思考模式时 Top-p 固定为 1；请将 Top-p 设为 1，或选择支持调整 Top-p 的模型"
        )
    if (
        config.thinking_mode == "disabled"
        and thinking_off_strategy(provider, model) is None
    ):
        errors.append(f"{provider}/{model} 无法由平台确认关闭思考模式")
    requested = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": (
            config.top_k
            if config.top_k_mode == "explicit"
            else ("disabled" if config.top_k_mode == "disabled" else None)
        ),
        "min_p": config.min_p,
        "typical_p": config.typical_p,
        "seed": config.seed,
        "presence_penalty": config.presence_penalty,
        "frequency_penalty": config.frequency_penalty,
        "repetition_penalty": config.repetition_penalty,
        "logit_bias": config.logit_bias,
        "n": config.n,
        "candidate_count": config.candidate_count,
        "best_of": config.best_of,
        "top_logprobs": config.top_logprobs,
        "max_output_tokens": config.max_output_tokens,
        "stop": config.stop,
        "thinking_mode": config.thinking_mode,
        "reasoning_effort": config.reasoning_effort,
        "thinking_budget": config.thinking_budget,
        "reasoning_summary": config.reasoning_summary,
        "response_format": config.response_format,
        "verbosity": config.verbosity,
        "truncation": config.truncation,
        "service_tier": config.service_tier,
        "prompt_cache_key": config.prompt_cache_key,
        "store": config.store,
    }
    if profile is None:
        if strict and any(value is not None for value in requested.values()):
            errors.append(f"未登记模型能力：{provider}/{model}")
        return errors
    for name, value in requested.items():
        if (
            name == "thinking_mode"
            and value == "disabled"
            and thinking_off_strategy(provider, model) == "intrinsic_off"
        ):
            continue
        if value is None or (name == "thinking_mode" and value == "provider_default"):
            continue
        rule = profile.parameters.get(name, ParameterRule())
        if rule.status == "unsupported":
            errors.append(f"{provider}/{model} 不支持 {name}：{rule.note}".rstrip("："))
        elif rule.status == "ignored" and strict:
            errors.append(f"{provider}/{model} 会忽略 {name}：{rule.note}".rstrip("："))
        elif rule.status == "unknown" and strict:
            errors.append(f"{provider}/{model} 的 {name} 尚未核实")
        elif (
            name == "reasoning_effort"
            and value is not None
            and rule.allowed_modes
            and str(value) not in rule.allowed_modes
        ):
            errors.append(
                f"{provider}/{model} 的 {name} 仅支持：{', '.join(rule.allowed_modes)}"
            )
    if config.thinking_mode == "disabled" and config.reasoning_effort is not None:
        errors.append("thinking_mode=disabled 时不能同时设置 reasoning_effort")
    if config.tools_enabled:
        errors.append("本项目批量运行器未配置 tools 定义，tools_enabled 不能启用")
    if provider == "gemini" and config.reasoning_effort and config.thinking_budget is not None:
        errors.append("Gemini不能同时设置 reasoning_effort 与 thinking_budget")
    return errors


def _merge_extra_body(config: GenerationConfig) -> Dict[str, Any]:
    return dict(config.extra_body or {})


def build_chat_request(
    provider: str,
    *,
    model: str,
    messages: Iterable[Dict[str, Any]],
    config: GenerationConfig,
    stream: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build a request body while preserving explicit values the model can use."""

    if config.tools_enabled:
        raise ValueError("本项目批量运行器未配置 tools 定义，tools_enabled 不能启用")
    config = force_thinking_off(config)
    if (
        str(provider).strip().casefold() == "deepseek"
        and config.top_p is not None
        and float(config.top_p) != 1.0
    ):
        raise ValueError(
            f"{provider}/{model} 关闭思考模式时 Top-p 固定为 1；请将 Top-p 设为 1，或选择支持调整 Top-p 的模型"
        )

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": config.stream if stream is None else bool(stream),
    }
    explicit = {
        "temperature": config.temperature,
        # DeepSeek documents Top-p as fixed at 1 when thinking is disabled.
        # Omit the field so the provider applies that fixed default, and reject
        # a non-1 user setting during validation instead of silently ignoring it.
        "top_p": (
            None
            if str(provider).strip().casefold() == "deepseek"
            and config.thinking_mode == "disabled"
            else config.top_p
        ),
        "seed": config.seed,
        "presence_penalty": config.presence_penalty,
        "frequency_penalty": config.frequency_penalty,
        "logit_bias": config.logit_bias,
        "n": config.n,
        "top_logprobs": config.top_logprobs,
        "stop": config.stop,
        "response_format": config.response_format,
        "service_tier": config.service_tier,
        "prompt_cache_key": config.prompt_cache_key,
        "truncation": config.truncation,
    }
    for name, value in explicit.items():
        if value is not None:
            kwargs[name] = value

    if config.max_output_tokens is not None:
        kwargs["max_completion_tokens" if provider == "openai" else "max_tokens"] = config.max_output_tokens

    extra = _merge_extra_body(config)
    top_k_rule = (get_capability(provider, model) or CapabilityProfile(provider, model, {})).parameters.get(
        "top_k", ParameterRule()
    )
    if config.top_k_mode != "provider_default":
        if top_k_rule.status in {"unsupported", "unknown"}:
            raise ValueError(
                f"{provider}/{model} 未登记 top_k 的{config.top_k_mode}映射"
            )
        if config.top_k_mode == "explicit":
            extra[top_k_rule.wire_field or "top_k"] = int(config.top_k)
        elif config.top_k_mode == "disabled":
            # Qwen's compatible endpoint uses null to disable top-k. Other
            # providers must explicitly register their own wire mapping.
            extra[top_k_rule.wire_field or "top_k"] = None
    if config.repetition_penalty is not None:
        extra["repetition_penalty"] = config.repetition_penalty
    if config.min_p is not None:
        extra["min_p"] = config.min_p
    if config.typical_p is not None:
        extra["typical_p"] = config.typical_p
    if config.candidate_count is not None:
        extra["candidate_count"] = config.candidate_count
    if config.best_of is not None:
        extra["best_of"] = config.best_of
    if config.verbosity is not None:
        extra["verbosity"] = config.verbosity
    strategy = thinking_off_strategy(provider, model)
    if strategy == "deepseek_toggle":
        extra["thinking"] = {"type": "disabled"}
    elif strategy == "qwen_toggle":
        extra["enable_thinking"] = False
    elif strategy == "reasoning_effort_none":
        kwargs["reasoning_effort"] = "none"
    elif strategy != "intrinsic_off":
        raise ValueError(f"{provider}/{model} 没有经过核实的关闭思考方式")
    if config.store is not None:
        kwargs["store"] = bool(config.store)
    if extra:
        kwargs["extra_body"] = extra
    return kwargs


def generation_config_for_legacy_temperature(
    temperature: Optional[float], generation_config: Optional[GenerationConfig]
) -> GenerationConfig:
    # A supplied GenerationConfig is authoritative.  In particular, an
    # explicit ``temperature=None`` is the natural/provider-default condition
    # and must not be silently replaced by the legacy UI temperature.
    if generation_config is None:
        return GenerationConfig(temperature=temperature)
    return generation_config
