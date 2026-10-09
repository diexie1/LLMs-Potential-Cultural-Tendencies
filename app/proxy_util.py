"""检测并应用系统代理（Windows / macOS / 环境变量）。"""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Optional
from urllib.request import getproxies


def _normalize_proxy(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    url = url.strip()
    if not url:
        return None
    if "://" not in url:
        url = "http://" + url
    return url


def _windows_system_proxy() -> Optional[str]:
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as key:
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if not enabled:
                return None
            server, _ = winreg.QueryValueEx(key, "ProxyServer")
            return _normalize_proxy(str(server).split(";")[0])
    except Exception:
        return None


def _macos_system_proxy() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["scutil", "--proxy"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        info: dict[str, str] = {}
        for line in out.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()

        if info.get("HTTPSEnable") == "1":
            host = info.get("HTTPSProxy")
            port = info.get("HTTPSPort")
            if host and port:
                return f"http://{host}:{port}"
        if info.get("HTTPEnable") == "1":
            host = info.get("HTTPProxy")
            port = info.get("HTTPPort")
            if host and port:
                return f"http://{host}:{port}"
        if info.get("SOCKSEnable") == "1":
            host = info.get("SOCKSProxy")
            port = info.get("SOCKSPort")
            if host and port:
                return f"socks5://{host}:{port}"
    except Exception:
        return None
    return None


def detect_system_proxy() -> Optional[str]:
    """
    优先级：
    1. 环境变量 HTTPS_PROXY / HTTP_PROXY / ALL_PROXY
    2. urllib.getproxies()（尊重系统设置）
    3. 平台专用读取（Windows 注册表 / macOS scutil）
    """
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        val = _normalize_proxy(os.environ.get(key))
        if val:
            return val

    try:
        proxies = getproxies()
        for key in ("https", "http", "all"):
            val = _normalize_proxy(proxies.get(key))
            if val:
                return val
    except Exception:
        pass

    system = platform.system().lower()
    if system == "windows":
        return _windows_system_proxy()
    if system == "darwin":
        return _macos_system_proxy()
    return None


def apply_proxy_to_env(proxy: Optional[str] = None) -> Optional[str]:
    """将检测到的代理写入当前进程环境，供 openai / requests 使用。"""
    proxy = proxy if proxy is not None else detect_system_proxy()
    if not proxy:
        return None
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ[key] = proxy
    return proxy


def requests_proxies(proxy: Optional[str] = None) -> Optional[dict]:
    proxy = proxy if proxy is not None else detect_system_proxy()
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}
