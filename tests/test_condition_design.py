from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.condition_design import (
    build_condition_descriptor,
    compare_condition_configs,
)
from app.experiment_presets import apply_preset
from app.generation_config import GenerationConfig, generation_config_for_legacy_temperature


class ConditionDesignTests(unittest.TestCase):
    def test_natural_none_is_not_replaced_by_legacy_temperature(self):
        config = GenerationConfig(temperature=None)
        self.assertIsNone(
            generation_config_for_legacy_temperature(0.7, config).temperature
        )
        self.assertEqual(
            build_condition_descriptor("natural", config)["interpretation"],
            "provider_default_requested",
        )

    def test_natural_and_controlled_requested_configs_are_distinguishable(self):
        main = apply_preset("free_response_v1")
        temperature_one = apply_preset("free_response_temperature_1_v1")
        comparison = compare_condition_configs(
            main["generation_config"], temperature_one["generation_config"]
        )
        self.assertFalse(comparison["requested_equivalent"])
        self.assertEqual(comparison["requested_difference_fields"], ["temperature"])

    def test_same_request_keeps_overlap_detectable(self):
        config = GenerationConfig(temperature=1.0, top_p=1.0)
        comparison = compare_condition_configs(config, config)
        self.assertTrue(comparison["requested_equivalent"])
        self.assertEqual(comparison["effective_equivalence"], "unknown")

    def test_controlled_label_without_values_is_not_called_explicit(self):
        descriptor = build_condition_descriptor(
            "controlled", GenerationConfig(temperature=None, top_p=None)
        )
        self.assertEqual(
            descriptor["interpretation"], "controlled_without_explicit_settings"
        )
        self.assertEqual(descriptor["explicit_fields"], [])


if __name__ == "__main__":
    unittest.main()
