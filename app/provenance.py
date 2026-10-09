"""运行、网络、账户和 HTTP 请求的可审计来源快照。

本模块只保存研究复现所需的身份信息，不保存 API key 或代理认证信息。
公网 IP 属于一次“经指定代理向第三方查询”的出口观测，不冒充供应商一定看到的
客户端地址；供应商是否提供请求 ID/区域等信息也会明确记录为未知。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit


SAFE_RESPONSE_HEADERS = {
    "x-request-id",
    "openai-processing-ms",
    "openai-version",
    "openai-organization",
    "openai-project",
    "retry-after",
    "server",
    "date",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    try:
        return str(value)
    except Exception:
        return repr(value)


def canonical_hash(value: Any) -> str:
    """Hash a JSON-safe value with stable key ordering."""

    canonical = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_project_root()),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        value = result.stdout.strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
        return None


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def build_software_snapshot() -> Dict[str, Any]:
    """Capture executable/runtime identity without requiring a Git checkout."""

    try:
        from . import __version__ as app_version
    except Exception:
        app_version = None
    executable = Path(sys.executable).resolve()
    snapshot = {
        "application_version": app_version,
        "git_commit": _git_commit(),
        "executable": str(executable),
        "executable_sha256": file_sha256(executable),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "timezone": datetime.now().astimezone().tzname(),
        "locale": os.environ.get("LANG") or os.environ.get("LC_ALL"),
        "sdk_versions": {
            name: version
            for name in ("openai", "httpx", "requests", "flask", "openpyxl")
            if (version := _package_version(name)) is not None
        },
        "captured_at_utc": now_utc(),
    }
    snapshot["snapshot_id"] = canonical_hash(snapshot)
    return snapshot


def sanitize_proxy(proxy: Optional[str]) -> Dict[str, Any]:
    """Return a displayable proxy identity with credentials removed."""

    raw = str(proxy or "").strip()
    if not raw:
        return {
            "configured": False,
            "display": None,
            "scheme": None,
            "host": None,
            "port": None,
            "credentials_present": False,
            "fingerprint": None,
        }
    normalized = raw if "://" in raw else "http://" + raw
    try:
        parts = urlsplit(normalized)
        scheme = parts.scheme.lower() or "http"
        host = parts.hostname
        port = parts.port
        host_display = f"[{host}]" if host and ":" in host else host
        display = f"{scheme}://{host_display or ''}"
        if port is not None:
            display += f":{port}"
        fingerprint = hashlib.sha256(display.encode("utf-8")).hexdigest()
        return {
            "configured": True,
            "display": display,
            "scheme": scheme,
            "host": host,
            "port": port,
            "credentials_present": bool(parts.username or parts.password),
            "fingerprint": fingerprint,
        }
    except ValueError:
        fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return {
            "configured": True,
            "display": "invalid-proxy-url",
            "scheme": None,
            "host": None,
            "port": None,
            "credentials_present": "@" in normalized,
            "fingerprint": fingerprint,
        }


def build_account_snapshot(provider: str, api_key: Optional[str]) -> Dict[str, Any]:
    """Identify a credential without persisting its secret value."""

    key = str(api_key or "")
    return {
        "provider": str(provider or ""),
        "credential_present": bool(key),
        "credential_fingerprint": (
            hashlib.sha256(key.encode("utf-8")).hexdigest() if key else None
        ),
    }


def build_network_snapshot(
    *,
    proxy: Optional[str] = None,
    include_public_ip: bool = True,
    timeout: float = 2.0,
) -> Dict[str, Any]:
    """Capture best-effort egress and proxy information.

    ``observed_public_ip`` is deliberately named as an observation because
    split tunnelling or destination-specific routing may make the provider's
    seen source address different from a third-party IP service.
    """

    if proxy is None:
        try:
            from . import proxy_util

            proxy = proxy_util.detect_system_proxy()
        except Exception:
            proxy = None
    info: Dict[str, Any] = {}
    if include_public_ip:
        try:
            from .net_info import detect_public_ip_info

            info = detect_public_ip_info(timeout=timeout)
        except Exception as exc:
            info = {"error": str(exc)}
    public_ip = info.get("ip")
    snapshot = {
        "snapshot_schema_version": "network-snapshot-v1",
        "observed_public_ip": public_ip,
        "public_ip_sha256": (
            hashlib.sha256(str(public_ip).encode("utf-8")).hexdigest()
            if public_ip
            else None
        ),
        "country": info.get("country"),
        "region": info.get("region"),
        "city": info.get("city"),
        "organization": info.get("org"),
        "lookup_source": info.get("source"),
        "lookup_error": info.get("error"),
        "proxy": sanitize_proxy(proxy),
        "observation_scope": "third_party_egress_observation",
        "observed_at_utc": now_utc(),
    }
    # Calculate before inserting the ID to avoid a self-referential hash.
    snapshot["snapshot_id"] = canonical_hash(snapshot)
    return snapshot


def compare_network_snapshots(
    before: Optional[Mapping[str, Any]], after: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Compare only observations that are available on both sides."""

    before = dict(before or {})
    after = dict(after or {})
    checks = {
        "public_ip": (
            before.get("public_ip_sha256"),
            after.get("public_ip_sha256"),
        ),
        "proxy": (
            (before.get("proxy") or {}).get("fingerprint"),
            (after.get("proxy") or {}).get("fingerprint"),
        ),
    }
    changed_fields = [
        name for name, (left, right) in checks.items() if left and right and left != right
    ]
    comparable = [name for name, (left, right) in checks.items() if left and right]
    return {
        "comparable": comparable,
        "changed_fields": changed_fields,
        "network_changed": bool(changed_fields),
        "status": "changed" if changed_fields else ("stable" if comparable else "unknown"),
    }


def safe_response_headers(headers: Any) -> Dict[str, str]:
    """Keep a small allowlist of non-secret response headers."""

    if headers is None:
        return {}
    try:
        items = headers.items()
    except AttributeError:
        return {}
    result: Dict[str, str] = {}
    for key, value in items:
        lower = str(key).lower()
        if lower in SAFE_RESPONSE_HEADERS:
            result[lower] = str(value)
    return result


def extract_request_id(value: Any) -> Optional[str]:
    """Extract common SDK/HTTP request IDs without assuming a provider shape."""

    for attr in ("_request_id", "request_id", "provider_request_id"):
        try:
            candidate = getattr(value, attr, None)
        except Exception:
            candidate = None
        if candidate:
            return str(candidate)
    headers = extract_response_headers(value)
    return headers.get("x-request-id")


def extract_response_headers(value: Any) -> Dict[str, str]:
    for attr in ("response", "_response", "http_response"):
        try:
            response = getattr(value, attr, None)
        except Exception:
            response = None
        headers = safe_response_headers(getattr(response, "headers", None))
        if headers:
            return headers
    return safe_response_headers(getattr(value, "headers", None))


def extract_error_metadata(exc: BaseException) -> Dict[str, Any]:
    request_id = extract_request_id(exc)
    headers = extract_response_headers(exc)
    if request_id is None:
        request_id = headers.get("x-request-id")
    result: Dict[str, Any] = {"response_headers": headers}
    if request_id:
        result["provider_request_id"] = request_id
    return result


def build_run_context(
    *,
    run_id: str,
    proxy: Optional[str],
    include_public_ip: bool = True,
) -> Dict[str, Any]:
    network_start = build_network_snapshot(
        proxy=proxy, include_public_ip=include_public_ip
    )
    return {
        "schema_version": "run-context-v1",
        "run_id": run_id,
        "software": build_software_snapshot(),
        "network_start": network_start,
        "network_start_snapshot_id": network_start.get("snapshot_id"),
        "created_at_utc": now_utc(),
    }
