"""各平台 API 调用：每次新建客户端，保证无记忆单轮对话。"""

from __future__ import annotations

from typing import Any, Dict, Generator, Iterable, List, Optional

import requests
from openai import OpenAI

from . import proxy_util
from .config import DEFAULT_API_KEYS


def deepseek_api(
    query: str,
    *,
    model: str = "deepseek-v4-pro",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant",
) -> str:
    cfg = DEFAULT_API_KEYS["deepseek"]
    client = OpenAI(
        api_key=api_key or cfg["api_key"],
        base_url=base_url or cfg["base_url"],
        http_client=_make_httpx_client(),
    )
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "stream": False,
        "temperature": temperature,
    }
    # deepseek-v4 / reasoner 系列可能支持 thinking；失败时回退普通调用
    try:
        response = client.chat.completions.create(
            **kwargs,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )
    except Exception:
        response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def openai_api(
    query: str,
    *,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant",
) -> str:
    cfg = DEFAULT_API_KEYS["openai"]
    client = OpenAI(
        api_key=api_key or cfg["api_key"],
        base_url=base_url or cfg["base_url"],
        http_client=_make_httpx_client(),
    )
    # 部分推理模型不接受 temperature
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            stream=False,
            temperature=temperature,
        )
    except Exception as first_err:
        err_text = str(first_err).lower()
        if "temperature" in err_text:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                stream=False,
            )
        else:
            raise
    return (response.choices[0].message.content or "").strip()


def qwen_api(
    query: str,
    *,
    model: str = "qwen-plus",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant",
) -> str:
    cfg = DEFAULT_API_KEYS["qwen"]
    client = OpenAI(
        api_key=api_key or cfg["api_key"],
        base_url=base_url or cfg["base_url"],
        http_client=_make_httpx_client(),
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        stream=False,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def gemini_api(
    query: str,
    *,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant",
) -> str:
    cfg = DEFAULT_API_KEYS["gemini"]
    client = OpenAI(
        api_key=api_key or cfg["api_key"],
        base_url=base_url or cfg["base_url"],
        http_client=_make_httpx_client(),
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        stream=False,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def _make_httpx_client():
    try:
        import httpx
    except ImportError:
        return None
    proxy = proxy_util.detect_system_proxy()
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
    temperature: float,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant",
) -> str:
    """每次调用独立创建连接，不携带历史对话。"""
    fn = PROVIDERS.get(provider)
    if fn is None:
        raise ValueError(f"未知 API 提供方: {provider}")
    return fn(
        query,
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        system_prompt=system_prompt,
    )


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


def list_models(provider: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> List[str]:
    """查询平台可用模型列表。"""
    from .config import DEFAULT_MODELS

    cfg = DEFAULT_API_KEYS.get(provider, {})
    key = api_key or cfg.get("api_key")
    url = (base_url or cfg.get("base_url") or "").rstrip("/")

    if provider in ("deepseek", "openai", "qwen", "gemini"):
        try:
            endpoint = f"{url}/models"
            headers = {"Authorization": f"Bearer {key}"}
            resp = requests.get(
                endpoint,
                headers=headers,
                proxies=proxy_util.requests_proxies(),
                timeout=30,
            )
            resp.raise_for_status()
            resp.encoding = "utf-8"
            data = resp.json().get("data", [])
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
                return names
        except Exception:
            pass

    return list(DEFAULT_MODELS.get(provider, []))


def _normalize_messages(
    messages: Iterable[Dict[str, Any]],
    *,
    system_prompt: str = "You are a helpful assistant",
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
    if not has_system:
        out.insert(0, {"role": "system", "content": system_prompt})
    return out


def _stream_openai_compatible(
    *,
    provider: str,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    api_key: Optional[str],
    base_url: Optional[str],
) -> Generator[str, None, None]:
    cfg = DEFAULT_API_KEYS[provider]
    client = OpenAI(
        api_key=api_key or cfg["api_key"],
        base_url=base_url or cfg["base_url"],
        http_client=_make_httpx_client(),
    )
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
    }
    try:
        stream = client.chat.completions.create(**kwargs)
    except Exception as first_err:
        err_text = str(first_err).lower()
        if "temperature" in err_text:
            kwargs.pop("temperature", None)
            stream = client.chat.completions.create(**kwargs)
        else:
            raise
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


def stream_chat(
    provider: str,
    messages: List[Dict[str, Any]],
    *,
    model: str,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant",
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
        )
        return
    raise ValueError(f"未知 API 提供方: {provider}")
