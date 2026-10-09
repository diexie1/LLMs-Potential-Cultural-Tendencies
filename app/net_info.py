"""公网 IP 与地理位置检测（经系统代理出口）。"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from . import proxy_util


def detect_public_ip_info(timeout: float = 8.0) -> Dict[str, Any]:
    """
    查询当前公网 IP 及大致位置。
    会走系统代理，因此显示的是实际出口 IP。
    """
    proxy_util.apply_proxy_to_env()
    proxies = proxy_util.requests_proxies()
    errors: list[str] = []

    # 多个免费接口兜底
    endpoints = [
        ("https://ipapi.co/json/", _parse_ipapi),
        ("https://ipinfo.io/json", _parse_ipinfo),
        ("http://ip-api.com/json/?lang=zh-CN", _parse_ip_api),
    ]
    for url, parser in endpoints:
        try:
            resp = requests.get(url, proxies=proxies, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            info = parser(data)
            if info.get("ip"):
                info["source"] = url
                return info
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    return {
        "ip": None,
        "location": "查询失败",
        "country": "",
        "region": "",
        "city": "",
        "org": "",
        "error": "; ".join(errors) if errors else "unknown",
    }


def _parse_ipapi(data: dict) -> Dict[str, Any]:
    if data.get("error"):
        raise RuntimeError(str(data.get("reason") or data.get("error")))
    city = data.get("city") or ""
    region = data.get("region") or data.get("region_code") or ""
    country = data.get("country_name") or data.get("country") or ""
    parts = [p for p in (country, region, city) if p]
    return {
        "ip": data.get("ip"),
        "location": " / ".join(parts) if parts else "未知位置",
        "country": country,
        "region": region,
        "city": city,
        "org": data.get("org") or "",
    }


def _parse_ipinfo(data: dict) -> Dict[str, Any]:
    city = data.get("city") or ""
    region = data.get("region") or ""
    country = data.get("country") or ""
    parts = [p for p in (country, region, city) if p]
    return {
        "ip": data.get("ip"),
        "location": " / ".join(parts) if parts else (data.get("loc") or "未知位置"),
        "country": country,
        "region": region,
        "city": city,
        "org": data.get("org") or "",
    }


def _parse_ip_api(data: dict) -> Dict[str, Any]:
    if data.get("status") != "success":
        raise RuntimeError(data.get("message") or "ip-api failed")
    city = data.get("city") or ""
    region = data.get("regionName") or ""
    country = data.get("country") or ""
    parts = [p for p in (country, region, city) if p]
    return {
        "ip": data.get("query"),
        "location": " / ".join(parts) if parts else "未知位置",
        "country": country,
        "region": region,
        "city": city,
        "org": data.get("isp") or data.get("org") or "",
    }


def network_status_text(ip_info: Optional[dict] = None, proxy: Optional[str] = None) -> str:
    if ip_info is None:
        ip_info = detect_public_ip_info()
    if proxy is None:
        proxy = proxy_util.detect_system_proxy()

    ip = ip_info.get("ip") or "未知"
    loc = ip_info.get("location") or "未知"
    org = ip_info.get("org") or ""
    org_part = f"（{org}）" if org else ""
    if proxy:
        from .provenance import sanitize_proxy

        proxy_display = sanitize_proxy(proxy).get("display") or "已配置（已隐藏）"
        proxy_part = f"系统代理: {proxy_display}"
    else:
        proxy_part = "系统代理: 未检测到（直连）"
    return f"公网 IP: {ip}  |  位置: {loc}{org_part}  |  {proxy_part}"
