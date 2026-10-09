"""各平台 API 调用：每次新建客户端，保证无记忆单轮对话。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import uuid
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

import requests
from openai import OpenAI

from . import proxy_util
from .config import DEFAULT_API_KEYS
from .generation_config import (
    GenerationConfig,
    build_chat_request,
    force_thinking_off,
    generation_config_for_legacy_temperature,
    validate_generation_config,
)
from .model_profiles import build_model_profile
from .provenance import (
    canonical_hash,
    extract_request_id,
    extract_response_headers,
    safe_response_headers,
)


# 北京时间没有夏令时，使用固定 +08:00 偏移可避免 Windows 环境缺少
# IANA 时区数据库时 ZoneInfo 初始化失败。
SHANGHAI_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")


def _user_content(query: str, image_urls: Optional[Sequence[str]] = None) -> Any:
    """Build OpenAI-compatible multimodal user content when worksheet images exist."""

    urls = [str(url).strip() for url in (image_urls or []) if str(url).strip()]
    if not urls:
        return query
    content: List[Dict[str, Any]] = [{"type": "text", "text": query}]
    content.extend(
        {"type": "image_url", "image_url": {"url": url}}
        for url in urls
    )
    return content


def _format_api_result(
    response: Any,
    *,
    provider: str,
    requested_model: str,
    base_url: Optional[str],
    return_metadata: bool,
    temperature: Optional[float] = 0.7,
    generation_config: Optional[GenerationConfig] = None,
    client_request_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
) -> Any:
    """Return text for legacy callers, or text plus provider metadata."""

    message = response.choices[0].message
    text = (getattr(message, "content", None) or "").strip()
    if not return_metadata:
        return text

    recorded_at_utc = datetime.now(timezone.utc)
    effective = force_thinking_off(
        generation_config_for_legacy_temperature(temperature, generation_config)
    )
    request_snapshot = build_chat_request(
        provider,
        model=requested_model,
        messages=[],
        config=effective,
        stream=False,
    )
    response_snapshot = _redact_reasoning_payload(_jsonable(response))
    metadata: Dict[str, Any] = {
        "provider": provider,
        "requested_model": requested_model,
        "returned_model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "client_request_id": client_request_id,
        "provider_request_id": extract_request_id(response),
        "attempt_id": attempt_id,
        "response_headers": extract_response_headers(response),
        "created": getattr(response, "created", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "endpoint": _safe_endpoint(base_url),
        "recorded_at_utc": recorded_at_utc.isoformat(),
        "recorded_at_local": recorded_at_utc.astimezone(SHANGHAI_TZ).isoformat(),
        "finish_reason": getattr(response.choices[0], "finish_reason", None),
        "usage": _jsonable(getattr(response, "usage", None)),
        "reasoning_content_present": bool(
            getattr(message, "reasoning_content", None)
        ),
    }
    # SDK response objects differ by provider. Keep only small, useful identity
    # fields and omit empty values so result files stay readable.
    metadata = {key: value for key, value in metadata.items() if value is not None}
    profile = build_model_profile(
        provider,
        requested_model,
        config=effective,
        request_snapshot=request_snapshot,
        response_snapshot=response_snapshot,
        returned_model=getattr(response, "model", None),
        model_metadata=metadata,
    )
    metadata["model_profile"] = profile
    return {
        "text": text,
        "model_metadata": metadata,
        "generation_config": effective.to_dict(),
        "request_snapshot": request_snapshot,
        "model_profile": profile,
        "response_snapshot": response_snapshot,
    }


def _jsonable(value: Any) -> Any:
    """Convert provider SDK objects to a JSON-safe snapshot without failing a run."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"))
        except TypeError:
            try:
                return _jsonable(model_dump())
            except Exception:
                pass
        except Exception:
            pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _jsonable(to_dict())
        except Exception:
            pass
    try:
        return {str(key): _jsonable(item) for key, item in vars(value).items()}
    except Exception:
        return str(value)


def _redact_reasoning_payload(value: Any) -> Any:
    """Keep response snapshots from storing hidden reasoning if a provider returns it."""

    hidden_fields = {
        "reasoning_content",
        "reasoning_summary",
        "thought",
        "thoughts",
        "thinking_content",
    }
    if isinstance(value, list):
        return [_redact_reasoning_payload(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _redact_reasoning_payload(item)
            for key, item in value.items()
            if str(key).casefold() not in hidden_fields
        }
    return value


def _safe_endpoint(base_url: Optional[str]) -> Optional[str]:
    """Keep endpoint scheme/host/path while dropping query, fragment and auth."""

    raw = str(base_url or "").strip()
    if not raw:
        return None
    try:
        parts = urlsplit(raw)
        if parts.scheme or parts.netloc:
            host = parts.hostname or ""
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            netloc = host
            try:
                if parts.port is not None:
                    netloc += f":{parts.port}"
            except ValueError:
                # If a custom URL has an invalid port, still do not return
                # the original user/password-bearing netloc.
                pass
            return urlunsplit((parts.scheme, netloc, parts.path.rstrip("/"), "", ""))
    except ValueError:
        pass
    return raw.rstrip("/")


def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _chat_completion(
    provider: str,
    *,
    query: str,
    model: str,
    temperature: Optional[float],
    api_key: Optional[str],
    base_url: Optional[str],
    system_prompt: str,
    image_urls: Optional[Sequence[str]],
    generation_config: Optional[GenerationConfig],
    client_request_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    proxy: Optional[str] = None,
) -> Any:
    """Use one request builder for all non-streaming providers.

    The caller decides whether a configuration is valid. This function never
    drops an explicit parameter and never retries with a different request body.
    """

    cfg = DEFAULT_API_KEYS[provider]
    effective = force_thinking_off(
        generation_config_for_legacy_temperature(temperature, generation_config)
    )
    errors = validate_generation_config(provider, model, effective, strict=False)
    if errors:
        raise ValueError("生成参数不兼容：" + "；".join(errors))
    client_kwargs: Dict[str, Any] = {
        "api_key": api_key or cfg["api_key"],
        "base_url": base_url or cfg["base_url"],
        "http_client": _make_httpx_client(proxy),
        "max_retries": 0,
    }
    if client_request_id:
        # OpenAI documents this header; compatible providers normally ignore
        # unknown request headers, while the local ID remains useful even if
        # a proxy strips it.
        client_kwargs["default_headers"] = {
            "X-Client-Request-Id": str(client_request_id)
        }
    client = OpenAI(
        **client_kwargs,
    )
    user_content = _user_content(query, image_urls)
    messages = [{"role": "user", "content": user_content}]
    if str(system_prompt or "").strip():
        messages.insert(0, {"role": "system", "content": system_prompt})
    kwargs = build_chat_request(
        provider,
        model=model,
        messages=messages,
        config=effective,
        stream=False,
    )
    try:
        return client.chat.completions.create(**kwargs)
    finally:
        _close_client(client)


def deepseek_api(
    query: str,
    *,
    model: str = "deepseek-v4-pro",
    temperature: Optional[float] = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: str = "",
    image_urls: Optional[Sequence[str]] = None,
    return_metadata: bool = False,
    generation_config: Optional[GenerationConfig] = None,
    client_request_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    proxy: Optional[str] = None,
) -> Any:
    cfg = DEFAULT_API_KEYS["deepseek"]
    response = _chat_completion(
        "deepseek",
        query=query,
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        system_prompt=system_prompt,
        image_urls=image_urls,
        generation_config=generation_config,
        client_request_id=client_request_id,
        attempt_id=attempt_id,
        proxy=proxy,
    )
    return _format_api_result(
        response,
        provider="deepseek",
        requested_model=model,
        base_url=base_url or cfg["base_url"],
        return_metadata=return_metadata,
        temperature=temperature,
        generation_config=generation_config,
        client_request_id=client_request_id,
        attempt_id=attempt_id,
    )


def openai_api(
    query: str,
    *,
    model: str = "gpt-4o",
    temperature: Optional[float] = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: str = "",
    image_urls: Optional[Sequence[str]] = None,
    return_metadata: bool = False,
    generation_config: Optional[GenerationConfig] = None,
    client_request_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    proxy: Optional[str] = None,
) -> Any:
    cfg = DEFAULT_API_KEYS["openai"]
    response = _chat_completion(
        "openai",
        query=query,
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        system_prompt=system_prompt,
        image_urls=image_urls,
        generation_config=generation_config,
        client_request_id=client_request_id,
        attempt_id=attempt_id,
        proxy=proxy,
    )
    return _format_api_result(
        response,
        provider="openai",
        requested_model=model,
        base_url=base_url or cfg["base_url"],
        return_metadata=return_metadata,
        temperature=temperature,
        generation_config=generation_config,
        client_request_id=client_request_id,
        attempt_id=attempt_id,
    )


def qwen_api(
    query: str,
    *,
    model: str = "qwen-plus",
    temperature: Optional[float] = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: str = "",
    image_urls: Optional[Sequence[str]] = None,
    return_metadata: bool = False,
    generation_config: Optional[GenerationConfig] = None,
    client_request_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    proxy: Optional[str] = None,
) -> Any:
    cfg = DEFAULT_API_KEYS["qwen"]
    response = _chat_completion(
        "qwen",
        query=query,
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        system_prompt=system_prompt,
        image_urls=image_urls,
        generation_config=generation_config,
        client_request_id=client_request_id,
        attempt_id=attempt_id,
        proxy=proxy,
    )
    return _format_api_result(
        response,
        provider="qwen",
        requested_model=model,
        base_url=base_url or cfg["base_url"],
        return_metadata=return_metadata,
        temperature=temperature,
        generation_config=generation_config,
        client_request_id=client_request_id,
        attempt_id=attempt_id,
    )


def gemini_api(
    query: str,
    *,
    model: str = "gemini-2.0-flash",
    temperature: Optional[float] = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: str = "",
    image_urls: Optional[Sequence[str]] = None,
    return_metadata: bool = False,
    generation_config: Optional[GenerationConfig] = None,
    client_request_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    proxy: Optional[str] = None,
) -> Any:
    cfg = DEFAULT_API_KEYS["gemini"]
    response = _chat_completion(
        "gemini",
        query=query,
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        system_prompt=system_prompt,
        image_urls=image_urls,
        generation_config=generation_config,
        client_request_id=client_request_id,
        attempt_id=attempt_id,
        proxy=proxy,
    )
    return _format_api_result(
        response,
        provider="gemini",
        requested_model=model,
        base_url=base_url or cfg["base_url"],
        return_metadata=return_metadata,
        temperature=temperature,
        generation_config=generation_config,
        client_request_id=client_request_id,
        attempt_id=attempt_id,
    )


def _make_httpx_client(proxy: Optional[str] = None):
    try:
        import httpx
    except ImportError:
        return None
    proxy = proxy if proxy is not None else proxy_util.detect_system_proxy()
    kwargs: Dict[str, Any] = {"timeout": 180.0}
    if proxy:
        kwargs["proxy"] = proxy
    return httpx.Client(**kwargs)


PROVIDERS = {
    "deepseek": deepseek_api,
    "openai": openai_api,
    "qwen": qwen_api,
    "gemini": gemini_api,
}


def call_api(
    provider: str,
    query: str,
    *,
    model: str,
    temperature: Optional[float],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: str = "",
    image_urls: Optional[Sequence[str]] = None,
    return_metadata: bool = False,
    generation_config: Optional[GenerationConfig] = None,
    client_request_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    proxy: Optional[str] = None,
) -> Any:
    """每次调用独立创建连接，不携带历史对话。

    ``generation_config`` is optional for backwards compatibility; formal
    runners should always pass the frozen configuration explicitly.
    """
    fn = PROVIDERS.get(provider)
    if fn is None:
        raise ValueError(f"未知 API 提供方: {provider}")
    if return_metadata and not client_request_id:
        client_request_id = f"direct-{uuid.uuid4().hex}"
    result = fn(
        query,
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        system_prompt=system_prompt,
        image_urls=image_urls,
        return_metadata=return_metadata,
        generation_config=generation_config,
        client_request_id=client_request_id,
        attempt_id=attempt_id,
        proxy=proxy,
    )
    if not return_metadata:
        return result
    effective = force_thinking_off(
        generation_config_for_legacy_temperature(temperature, generation_config)
    )
    # Reconstruct the request-side settings without credentials. The batch
    # runner supplies its richer prompt/image snapshot separately; this keeps
    # direct API/chat callers auditable too.
    request_messages = [
        {"role": "user", "content": _user_content(query, image_urls)},
    ]
    if str(system_prompt or "").strip():
        request_messages.insert(0, {"role": "system", "content": system_prompt})
    request_snapshot = build_chat_request(
        provider,
        model=model,
        messages=request_messages,
        config=effective,
        stream=False,
    )
    # Add request-side identity even if an SDK did not return model metadata.
    if not isinstance(result, dict) or "text" not in result:
        recorded_at_utc = datetime.now(timezone.utc)
        profile = build_model_profile(
            provider,
            model,
            config=effective,
            request_snapshot=request_snapshot,
        )
        return {
            "text": str(result or ""),
            "model_metadata": {
                "provider": provider,
                "requested_model": model,
                "client_request_id": client_request_id,
                "attempt_id": attempt_id,
                "provider_request_id": None,
                "response_headers": {},
                "endpoint": _safe_endpoint(base_url),
                "recorded_at_utc": recorded_at_utc.isoformat(),
                "recorded_at_local": recorded_at_utc.astimezone(SHANGHAI_TZ).isoformat(),
                "model_profile": profile,
            },
            "generation_config": effective.to_dict(),
            "request_snapshot": request_snapshot,
            "model_profile": profile,
            "response_snapshot": None,
        }
    result.setdefault("model_metadata", {})
    result["model_metadata"].setdefault("provider", provider)
    result["model_metadata"].setdefault("requested_model", model)
    result["model_metadata"].setdefault(
        "endpoint", _safe_endpoint(base_url)
    )
    if client_request_id:
        result["model_metadata"].setdefault("client_request_id", client_request_id)
    if attempt_id:
        result["model_metadata"].setdefault("attempt_id", attempt_id)
    profile = build_model_profile(
        provider,
        model,
        config=effective,
        request_snapshot=request_snapshot,
        response_snapshot=result.get("response_snapshot"),
        returned_model=result["model_metadata"].get("returned_model"),
        model_metadata=result["model_metadata"],
    )
    result["generation_config"] = effective.to_dict()
    result["request_snapshot"] = request_snapshot
    result["model_profile"] = profile
    result["model_metadata"]["model_profile"] = profile
    return result


def _normalize_model_id(provider: str, model_id: str) -> str:
    """统一模型 ID；Gemini OpenAI 兼容接口常返回 models/xxx。"""
    name = str(model_id or "").strip()
    if provider == "gemini" and name.startswith("models/"):
        name = name[len("models/") :]
    return name


def _model_belongs_to_provider(provider: str, model_id: str) -> bool:
    """按提供方过滤远程 /models 结果，避免 Base URL 配错时串台。"""
    name = model_id.lower()
    if provider == "deepseek":
        return "deepseek" in name
    if provider == "qwen":
        return "qwen" in name
    if provider == "gemini":
        return name.startswith("gemini")
    return True


def fetch_model_catalog(
    provider: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    proxy: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch a safe, hashable model-catalog snapshot for one provider."""

    from .config import DEFAULT_MODELS

    cfg = DEFAULT_API_KEYS.get(provider, {})
    key = api_key or cfg.get("api_key")
    url = (base_url or cfg.get("base_url") or "").rstrip("/")
    endpoint = f"{url}/models"
    error: Optional[str] = None

    if not key:
        error = "credential_missing"
    elif provider in ("deepseek", "openai", "qwen", "gemini"):
        try:
            headers = {"Authorization": f"Bearer {key}"}
            resp = requests.get(
                endpoint,
                headers=headers,
                proxies=proxy_util.requests_proxies(proxy),
                timeout=8,
            )
            resp.raise_for_status()
            resp.encoding = "utf-8"
            body = resp.json()
            data = body.get("data", []) if isinstance(body, dict) else []
            names = []
            for item in data:
                raw = item.get("id") if isinstance(item, dict) else None
                if not raw:
                    continue
                mid = _normalize_model_id(provider, raw)
                if mid and _model_belongs_to_provider(provider, mid):
                    names.append(mid)
            names = sorted(set(names), key=str.lower)
            if names:
                return {
                    "schema_version": "model-catalog-v1",
                    "provider": provider,
                    "source": "remote",
                    "endpoint": _safe_endpoint(endpoint),
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "status_code": resp.status_code,
                    "provider_request_id": resp.headers.get("x-request-id"),
                    "response_headers": safe_response_headers(resp.headers),
                    "model_ids": names,
                    "response_body": _jsonable(body),
                    "response_body_sha256": canonical_hash(body),
                }
        except Exception as exc:
            error = str(exc)
    else:
        error = "provider_not_supported"

    fallback = list(DEFAULT_MODELS.get(provider, []))
    return {
        "schema_version": "model-catalog-v1",
        "provider": provider,
        "source": "local_fallback",
        "endpoint": _safe_endpoint(endpoint),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "status_code": None,
        "provider_request_id": None,
        "response_headers": {},
        "model_ids": fallback,
        "response_body": None,
        "response_body_sha256": None,
        "error": error,
    }


def list_models(
    provider: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    proxy: Optional[str] = None,
) -> List[str]:
    """查询平台可用模型列表，同时保留详细快照的内部接口。"""

    return list(
        fetch_model_catalog(
            provider, api_key=api_key, base_url=base_url, proxy=proxy
        ).get("model_ids")
        or []
    )


def _normalize_messages(
    messages: Iterable[Dict[str, Any]],
    *,
    system_prompt: str = "",
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    has_system = False
    for item in messages:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "")
        if role not in ("system", "user", "assistant") or not content.strip():
            continue
        if role == "system":
            has_system = True
        out.append({"role": role, "content": content})
    if not has_system and str(system_prompt or "").strip():
        out.insert(0, {"role": "system", "content": system_prompt})
    return out


def _stream_openai_compatible(
    *,
    provider: str,
    messages: List[Dict[str, str]],
    model: str,
    temperature: Optional[float],
    api_key: Optional[str],
    base_url: Optional[str],
    generation_config: Optional[GenerationConfig] = None,
    client_request_id: Optional[str] = None,
    proxy: Optional[str] = None,
) -> Generator[str, None, None]:
    cfg = DEFAULT_API_KEYS[provider]
    effective = force_thinking_off(
        generation_config_for_legacy_temperature(temperature, generation_config)
    )
    errors = validate_generation_config(provider, model, effective, strict=False)
    if errors:
        raise ValueError("生成参数不兼容：" + "；".join(errors))
    client_kwargs: Dict[str, Any] = {
        "api_key": api_key or cfg["api_key"],
        "base_url": base_url or cfg["base_url"],
        "http_client": _make_httpx_client(proxy),
        "max_retries": 0,
    }
    if client_request_id:
        client_kwargs["default_headers"] = {
            "X-Client-Request-Id": str(client_request_id)
        }
    client = OpenAI(**client_kwargs)
    kwargs = build_chat_request(
        provider,
        model=model,
        messages=messages,
        config=effective,
        stream=True,
    )
    try:
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            try:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
            except Exception:
                text = ""
            if text:
                yield text
    finally:
        _close_client(client)


def stream_chat(
    provider: str,
    messages: List[Dict[str, Any]],
    *,
    model: str,
    temperature: Optional[float] = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: str = "",
    generation_config: Optional[GenerationConfig] = None,
    client_request_id: Optional[str] = None,
    proxy: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream assistant tokens for multi-turn chat (OpenAI-compatible APIs)."""
    normalized = _normalize_messages(messages, system_prompt=system_prompt)
    if provider in ("deepseek", "openai", "qwen", "gemini"):
        yield from _stream_openai_compatible(
            provider=provider,
            messages=normalized,
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            generation_config=generation_config,
            client_request_id=client_request_id or uuid.uuid4().hex,
            proxy=proxy,
        )
        return
    raise ValueError(f"未知 API 提供方: {provider}")
