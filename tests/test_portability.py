"""Offline regression checks for released startup and path handling."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import __version__
from app.startup import application_identity, find_existing_server
from app.webapp import app
from build_portable_zip import project_files, validate_runtime


class PortabilityTests(unittest.TestCase):
    def test_version_and_health_match_without_local_configuration(self):
        self.assertEqual((ROOT / "VERSION").read_text(encoding="utf-8").strip(), __version__)
        with patch("app.webapp.load_user_config", side_effect=AssertionError("Must not read credentials")):
            client = app.test_client()
            health = client.get("/api/health")
            self.assertEqual(health.get_json(), {**application_identity(), "pid": os.getpid()})
            self.assertEqual(client.get("/api/meta").get_json()["version"], __version__)
            for page in ("/", "/console", "/chat"):
                self.assertIn(f"v{__version__}", client.get(page).get_data(as_text=True))

    def test_different_directories_have_different_identities(self):
        self.assertNotEqual(application_identity(ROOT)["workspace_id"],
                            application_identity(ROOT / "another copy")["workspace_id"])

    def test_port_guard_reuses_only_matching_application_version_and_directory(self):
        own = application_identity()
        wrong = [{}, {**own, "version": "old"}, {**own, "workspace_id": "other-copy"}, {**own, "app_id": "other-app"}]
        for payload in wrong:
            with self.subTest(payload=payload), patch("app.startup.build_opener") as builder:
                builder.return_value.open.side_effect = [io.BytesIO(json.dumps(payload).encode()),
                                                        io.BytesIO(json.dumps(own).encode())]
                self.assertEqual(find_existing_server("127.0.0.1", 8765, span=2), "http://127.0.0.1:8766/")
        with patch("app.startup.build_opener") as builder:
            builder.return_value.open.side_effect = OSError("not listening")
            self.assertIsNone(find_existing_server("127.0.0.1", 8765, span=2))

    def test_runtime_import_paths_must_be_relative(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "runtime").mkdir()
            pth = root / "runtime/python312._pth"
            pth.write_text("python312.zip\n.\nLib\\site-packages\nimport site\n", encoding="utf-8")
            validate_runtime(root)
            for invalid in ("C:\\old machine\\packages", "\\\\server\\share", "\\rooted", "/absolute"):
                pth.write_text(invalid, encoding="utf-8")
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    validate_runtime(root)

    def test_bundle_excludes_absolute_path_entry_points_and_caches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("app/webapp.py", "runtime/python.exe", "runtime/Scripts/pip.exe",
                         "runtime/get-pip.py", "app/__pycache__/test.pyc"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            names = {path.as_posix() for path in project_files(root)}
            self.assertIn("runtime/python.exe", names)
            self.assertNotIn("runtime/Scripts/pip.exe", names)
            self.assertNotIn("runtime/get-pip.py", names)
            self.assertNotIn("app/__pycache__/test.pyc", names)
            self.assertNotIn("user_config.json", names)


if __name__ == "__main__":
    unittest.main()
