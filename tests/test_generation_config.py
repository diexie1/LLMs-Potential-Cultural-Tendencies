from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.generation_config import (  # noqa: E402
    GenerationConfig,
    build_chat_request,
    get_capability,
    validate_generation_config,
)
from app import __version__ as APP_VERSION  # noqa: E402
from app.api_client import _safe_endpoint, deepseek_api  # noqa: E402
from app.model_profiles import build_model_profile  # noqa: E402
from app.net_info import network_status_text  # noqa: E402
from app.provenance import compare_network_snapshots, sanitize_proxy  # noqa: E402
from app.runner import (  # noqa: E402
    BatchRunner,
    RunnerConfig,
    ScaleRunConfig,
    _manifest_hash,
    normalize_sample_mode,
)
from app.webapp import _as_bool, app  # noqa: E402
from app.scale_loader import ScaleFile, ScaleSheet  # noqa: E402
from app.experiment_presets import apply_preset  # noqa: E402


class GenerationConfigTests(unittest.TestCase):
    def test_trial_policy_is_fixed_and_legacy_valid_is_migrated(self):
        self.assertEqual(normalize_sample_mode("planned"), "planned")
        self.assertEqual(normalize_sample_mode("valid"), "planned")
        self.assertEqual(RunnerConfig(sample_mode="valid").sample_mode, "planned")
        with self.assertRaises(ValueError):
            normalize_sample_mode("unknown")

    def test_web_boolean_normalization_preserves_false_string(self):
        self.assertFalse(_as_bool("false", True))
        self.assertTrue(_as_bool("true", False))

    def test_web_config_exposes_provenance_defaults(self):
        response = app.test_client().get("/api/config")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("capture_network", data)
        self.assertIn("capture_model_catalog", data)
        self.assertIn("network_guard", data)
        self.assertIn("order_strategy", data)

    def test_meta_and_network_endpoints_return_json(self):
        client = app.test_client()
        with patch("app.webapp.proxy_util.detect_system_proxy", return_value=None), patch(
            "app.webapp.detect_public_ip_info",
            return_value={
                "ip": "203.0.113.10",
                "location": "测试位置",
                "org": "测试运营商",
            },
        ):
            meta = client.get("/api/meta")
            network = client.get("/api/network")
        self.assertEqual(meta.status_code, 200)
        self.assertEqual(network.status_code, 200)
        self.assertEqual(network.get_json()["ip"], "203.0.113.10")

    def test_experiment_preset_endpoint_and_main_analysis_variants(self):
        client = app.test_client()
        listing = client.get("/api/presets")
        self.assertEqual(listing.status_code, 200)
        payload = listing.get_json()
        self.assertEqual(payload["schema_version"], "experiment-preset-v2")
        self.assertEqual(
            {item["id"] for item in payload["presets"]},
            {
                "free_response_v1",
                "free_response_temperature_1_v1",
                "free_response_scores_only_v1",
                "china_identity_v1",
                "usa_identity_v1",
            },
        )
        self.assertEqual(
            {guide["id"] for guide in payload["parameter_guide"]},
            {"temperature", "top_p", "thinking_mode", "concurrency", "scale_concurrency", "prompt_condition"},
        )

        response = client.post(
            "/api/presets/apply",
            json={"preset_id": "free_response_temperature_1_v1"},
        )
        self.assertEqual(response.status_code, 200)
        preset = response.get_json()["preset"]
        generation = preset["generation_config"]
        execution = preset["execution_config"]
        self.assertEqual(generation["temperature"], 1.0)
        self.assertEqual(generation["top_p"], 1)
        self.assertEqual(generation["top_k_mode"], "provider_default")
        self.assertEqual(generation["thinking_mode"], "disabled")
        self.assertIsNone(generation["max_output_tokens"])
        self.assertEqual(preset["prompt_config"], {"contract": "unconstrained", "cultural_identity": "none"})
        self.assertEqual(execution["batch_en"], 100)
        self.assertEqual(execution["batch_ch"], 100)
        self.assertEqual(execution["sample_mode"], "planned")
        self.assertEqual(execution["random_seed"], 20260911)

    def test_unknown_experiment_preset_is_rejected(self):
        response = app.test_client().post(
            "/api/presets/apply",
            json={"preset_id": "does-not-exist"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("未知实验预设", response.get_json()["error"])

    def test_provenance_identity_and_alias_status_are_explicit(self):
        cfg = GenerationConfig(temperature=0, store=False)
        request = build_chat_request(
            "deepseek",
            model="deepseek-flash",
            messages=[],
            config=cfg,
        )
        profile = build_model_profile(
            "deepseek",
            "deepseek-flash",
            config=cfg,
            request_snapshot=request,
            returned_model="deepseek-flash",
            model_metadata={
                "endpoint": "https://api.deepseek.com",
                "client_request_id": "client-1",
                "provider_request_id": "provider-1",
                "attempt_id": "attempt-1",
            },
        )
        identity = profile["identity"]
        self.assertEqual(identity["identity_status"], "alias_or_snapshot_unverified")
        self.assertEqual(identity["endpoint"], "https://api.deepseek.com")
        self.assertEqual(identity["provider_request_id"], "provider-1")

    def test_proxy_credentials_are_not_in_snapshot_display(self):
        snapshot = sanitize_proxy("http://user:secret@127.0.0.1:7897")
        self.assertTrue(snapshot["credentials_present"])
        self.assertNotIn("user", snapshot["display"])
        self.assertNotIn("secret", snapshot["display"])
        self.assertEqual(snapshot["display"], "http://127.0.0.1:7897")

    def test_endpoint_credentials_and_query_are_not_recorded(self):
        self.assertEqual(
            _safe_endpoint("https://user:secret@example.test:8443/v1?token=abc#x"),
            "https://example.test:8443/v1",
        )

    def test_network_status_text_never_displays_proxy_credentials(self):
        text = network_status_text(
            {"ip": "198.51.100.1"}, "http://user:secret@127.0.0.1:7897"
        )
        self.assertIn("127.0.0.1:7897", text)
        self.assertNotIn("user", text)
        self.assertNotIn("secret", text)

    def test_network_comparison_distinguishes_unknown_and_changed(self):
        unknown = compare_network_snapshots({"proxy": {}}, {"proxy": {}})
        self.assertEqual(unknown["status"], "unknown")
        changed = compare_network_snapshots(
            {
                "public_ip_sha256": "a",
                "proxy": {"fingerprint": "p"},
            },
            {
                "public_ip_sha256": "b",
                "proxy": {"fingerprint": "p"},
            },
        )
        self.assertTrue(changed["network_changed"])
        self.assertEqual(changed["changed_fields"], ["public_ip"])

    def test_zero_and_false_are_preserved(self):
        cfg = GenerationConfig.from_mapping(
            {
                "temperature": 0,
                "top_p": 1,
                "thinking_mode": "disabled",
                "store": False,
            }
        )
        self.assertEqual(cfg.temperature, 0)
        self.assertEqual(cfg.top_p, 1)
        self.assertFalse(cfg.store)
        body = build_chat_request(
            "deepseek",
            model="deepseek-flash",
            messages=[{"role": "user", "content": "x"}],
            config=cfg,
        )
        self.assertEqual(body["temperature"], 0)
        self.assertNotIn("top_p", body)
        self.assertEqual(body["store"], False)
        self.assertEqual(body["extra_body"]["thinking"], {"type": "disabled"})

    def test_top_k_requires_registered_mapping(self):
        cfg = GenerationConfig(top_k=20, top_k_mode="explicit")
        with self.assertRaises(ValueError):
            build_chat_request(
                "deepseek",
                model="deepseek-flash",
                messages=[],
                config=cfg,
            )

        body = build_chat_request(
            "qwen",
            model="qwen-plus",
            messages=[],
            config=cfg,
        )
        self.assertEqual(body["extra_body"]["top_k"], 20)

    def test_tools_cannot_be_enabled_without_tool_definitions(self):
        cfg = GenerationConfig(tools_enabled=True)
        self.assertTrue(
            any("tools 定义" in error for error in validate_generation_config(
                "deepseek", "deepseek-flash", cfg
            ))
        )
        with self.assertRaises(ValueError):
            build_chat_request(
                "deepseek",
                model="deepseek-flash",
                messages=[],
                config=cfg,
            )

    def test_legacy_thinking_setting_is_forced_off_and_temperature_retained(self):
        cfg = GenerationConfig(temperature=0, thinking_mode="enabled")
        self.assertEqual(
            validate_generation_config("deepseek", "deepseek-flash", cfg), []
        )
        body = build_chat_request(
            "deepseek",
            model="deepseek-flash",
            messages=[],
            config=cfg,
        )
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["extra_body"]["thinking"], {"type": "disabled"})

    def test_strict_mode_rejects_unverified_model(self):
        cfg = GenerationConfig(temperature=0)
        errors = validate_generation_config(
            "qwen", "new-model-snapshot", cfg, strict=True
        )
        self.assertTrue(any("未登记模型能力" in error for error in errors))

    def test_qwen37_flash_natural_defaults_pass_strict_validation(self):
        cfg = GenerationConfig(temperature=None, thinking_mode="provider_default")
        profile = get_capability("qwen", "qwen3.7-flash")

        self.assertIsNotNone(profile)
        self.assertEqual(profile.as_of, "2026-09-13")
        self.assertIn("Alibaba Cloud", profile.source)
        self.assertEqual(
            validate_generation_config("qwen", "qwen3.7-flash", cfg, strict=True),
            [],
        )

        body = build_chat_request(
            "qwen",
            model="qwen3.7-flash",
            messages=[{"role": "user", "content": "test"}],
            config=cfg,
        )
        self.assertNotIn("temperature", body)
        self.assertEqual(body["extra_body"], {"enable_thinking": False})

    def test_qwen37_flash_verified_parameter_mapping(self):
        cfg = GenerationConfig(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            top_k_mode="explicit",
            seed=123,
            thinking_mode="disabled",
        )
        self.assertEqual(
            validate_generation_config("qwen", "qwen3.7-flash", cfg, strict=True),
            [],
        )
        body = build_chat_request(
            "qwen",
            model="qwen3.7-flash",
            messages=[],
            config=cfg,
        )
        self.assertEqual(body["temperature"], 0.6)
        self.assertEqual(body["top_p"], 0.95)
        self.assertEqual(body["seed"], 123)
        self.assertEqual(body["extra_body"]["top_k"], 20)
        self.assertIs(body["extra_body"]["enable_thinking"], False)

    def test_qwen37_flash_rejects_unverified_parameters(self):
        for config, expected in (
            (GenerationConfig(n=2), "不支持 n"),
            (GenerationConfig(reasoning_effort="high"), "不支持 reasoning_effort"),
        ):
            with self.subTest(parameter=expected):
                errors = validate_generation_config(
                    "qwen", "qwen3.7-flash", config, strict=True
                )
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_effort_and_budget_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            GenerationConfig(reasoning_effort="high", thinking_budget=100)

    def test_provider_wrapper_does_not_retry_with_a_different_body(self):
        message = SimpleNamespace(content='{"answers":[]}', reasoning_content="")
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            model="deepseek-flash",
            id="response-1",
            _request_id="provider-request-1",
            created=1,
            usage=SimpleNamespace(total_tokens=3),
        )
        config = GenerationConfig(temperature=0, thinking_mode="disabled")
        with patch("app.api_client.OpenAI") as client_cls:
            client_cls.return_value.chat.completions.create.return_value = response
            result = deepseek_api(
                "prompt",
                model="deepseek-flash",
                api_key="test",
                generation_config=config,
                return_metadata=True,
                client_request_id="client-request-1",
            )
        client_cls.return_value.chat.completions.create.assert_called_once()
        body = client_cls.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["extra_body"]["thinking"], {"type": "disabled"})
        self.assertEqual(result["model_metadata"]["finish_reason"], "stop")
        self.assertEqual(
            result["model_metadata"]["provider_request_id"], "provider-request-1"
        )
        self.assertEqual(
            result["model_metadata"]["client_request_id"], "client-request-1"
        )
        self.assertIn("response_snapshot", result)
        self.assertIn("model_profile", result)
        self.assertEqual(
            result["model_profile"]["parameters"]["temperature"]["resolution"],
            "explicit_request",
        )

    def test_model_profile_distinguishes_documented_and_observed_defaults(self):
        cfg = GenerationConfig(temperature=0, top_p=None)
        request = build_chat_request(
            "deepseek",
            model="deepseek-flash",
            messages=[],
            config=cfg,
        )
        profile = build_model_profile(
            "deepseek",
            "deepseek-flash",
            config=cfg,
            request_snapshot=request,
        )
        self.assertEqual(
            profile["parameters"]["temperature"]["resolution"],
            "explicit_request",
        )
        self.assertEqual(
            profile["parameters"]["top_p"]["resolution"],
            "documented_default_not_observed",
        )
        self.assertEqual(profile["parameters"]["top_p"]["effective_value"], 1)
        self.assertFalse(profile["parameters"]["top_p"]["sent"])

        observed = build_model_profile(
            "gemini",
            "gemini-2.5-flash",
            config=GenerationConfig(temperature=None, top_p=None),
            request_snapshot={"model": "gemini-2.5-flash", "messages": [], "stream": False},
            response_snapshot={"generationConfig": {"temperature": 0.8, "topP": 0.9}},
        )
        self.assertEqual(
            observed["parameters"]["temperature"]["resolution"],
            "observed_from_response",
        )
        self.assertEqual(observed["parameters"]["top_p"]["effective_value"], 0.9)

    def test_runner_writes_manifest_without_replacement_trials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sheet = ScaleSheet(
                language="ch",
                title="fixture",
                instruction="fixture",
                questions=["选择 A 或 B"],
            )
            scale = ScaleFile(path=root / "fixture.xlsx", name="fixture", ch=sheet)
            config = RunnerConfig(
                results_root=root / "results",
                logs_root=root / "logs",
                max_retries=1,
                retry_delay=0,
                concurrency=2,
                max_group_attempts=2,
                capture_network=False,
                capture_model_catalog=False,
            )
            config.scale_cfgs[scale.name] = ScaleRunConfig(
                scale_name=scale.name,
                ch_count=1,
                shuffle_items=False,
            )
            runner = BatchRunner([scale], config)
            # A transport exception remains retryable. A model response such
            # as "not json" is now preserved as a formal unparsed trial.
            with patch("app.runner.call_api", side_effect=TimeoutError("network timeout")) as call:
                runner.run()

            manifests = list((root / "logs").glob("*.manifest.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_version"], "run-manifest-v1")
            self.assertEqual(manifest["application_version"], APP_VERSION)
            self.assertEqual(manifest["tasks"][0]["count"], 1)
            self.assertIn("model_profiles", manifest)
            self.assertIn("deepseek/deepseek-v4-pro", manifest["model_profiles"])
            self.assertIn("run_context", manifest)
            self.assertIn("software", manifest["run_context"])
            self.assertIn("account_snapshots", manifest)
            self.assertTrue(manifest["account_snapshots"]["deepseek"]["snapshot_id"])
            self.assertTrue(
                manifest["run_context"]["network_start"]["snapshot_id"]
            )
            self.assertEqual(manifest["model_catalogs"], {})
            self.assertEqual(manifest["status"], "partial")
            self.assertEqual(manifest["manifest_hash"], _manifest_hash(manifest))
            # One configured trial means one submitted trial. A transport
            # failure is recorded as an attempt, but it never creates a
            # replacement trial.
            self.assertEqual(call.call_count, 1)
            attempts = list((root / "results").rglob("trial_*_attempt_*.json"))
            self.assertEqual(len(attempts), 1)
            attempt = json.loads(attempts[0].read_text(encoding="utf-8"))
            self.assertTrue(attempt["attempt_id"])
            self.assertTrue(attempt["client_request_id"])
            self.assertEqual(len(attempt["request_sha256"]), 64)
            self.assertFalse(list((root / "results").rglob("order_*.json")))
            plans = list((root / "results").rglob("plan.jsonl"))
            self.assertEqual(len(plans), 1)
            plan_rows = [json.loads(line) for line in plans[0].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(plan_rows), 1)
            self.assertEqual(plan_rows[0]["trial_id"], "fixture/ch/0001")
            self.assertIsNone(plan_rows[0]["order_seed"])
            self.assertIsNotNone(manifest["random_seed"])

    def test_order_seed_is_stable_and_planned_mode_does_not_replace_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sheet = ScaleSheet(
                language="ch",
                title="fixture",
                instruction="fixture",
                questions=["A", "B", "C"],
                default_shuffle=True,
                shuffle_supported=True,
            )
            scale = ScaleFile(path=root / "fixture.xlsx", name="fixture", ch=sheet)
            config = RunnerConfig(
                results_root=root / "results",
                logs_root=root / "logs",
                max_retries=1,
                retry_delay=0,
                concurrency=2,
                sample_mode="planned",
                random_seed=42,
                capture_network=False,
                capture_model_catalog=False,
            )
            config.scale_cfgs[scale.name] = ScaleRunConfig(
                scale_name=scale.name, ch_count=3, shuffle_items=True
            )
            runner = BatchRunner([scale], config)
            with patch("app.runner.call_api", return_value="not json") as call:
                runner.run()
            self.assertEqual(call.call_count, 3)
            plan = next((root / "results").rglob("plan.jsonl"))
            rows = [json.loads(line) for line in plan.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["order_id"] for row in rows], [1, 2, 3])
            self.assertEqual(len({row["order_seed"] for row in rows}), 3)


if __name__ == "__main__":
    unittest.main()
