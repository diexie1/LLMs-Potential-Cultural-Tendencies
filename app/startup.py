"""Identify this local platform without opening another application's port."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

from . import __version__
from .config import ROOT

APP_ID = "llm-cultural-tendencies"


def application_identity(root: Path = ROOT) -> dict:
    location = str(root.resolve())
    if os.name == "nt":
        location = location.casefold()
    return {"app_id": APP_ID, "version": __version__,
            "workspace_id": hashlib.sha256(location.encode("utf-8")).hexdigest()}


def find_existing_server(host: str, port: int, span: int = 20) -> str | None:
    expected = application_identity()
    opener = build_opener(ProxyHandler({}))
    for candidate in range(port, port + span):
        url = f"http://{host}:{candidate}"
        try:
            with opener.open(url + "/api/health", timeout=0.15) as response:
                identity = json.loads(response.read(4096))
            if isinstance(identity, dict) and all(identity.get(k) == v for k, v in expected.items()):
                return url + "/"
        except (OSError, ValueError):
            continue
    return None
