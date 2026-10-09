from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.prompting import (  # noqa: E402
    build_display_plan,
    build_prompt,
    parse_free_answers,
    parse_profile_answers,
)
from app.scale_loader import ScaleImage, ScaleSheet  # noqa: E402
from app.scale_profiles import (  # noqa: E402
    ProfileBuildError,
    SCALE_NAME_PROFILES,
    TYPED_FLAT_SPECS,
    build_profile_layout,
    identify_profile,
)


def _typed_sheet(profile_id: str, language: str = "ch") -> ScaleSheet:
    spec = TYPED_FLAT_SPECS[profile_id]
    cells = ["Fixture", "General instruction."] + [
        f"{index}. Fixture item {index}." for index in range(1, len(spec.rules) + 1)
    ]
    layout = build_profile_layout(profile_id, cells, language, [])
    return ScaleSheet(
        language=language,
        title="Fixture",
        instruction=layout.instruction,
        questions=[slot.question for block in layout.blocks for slot in block.slots],
        profile_id=layout.profile_id,
        profile_version=layout.profile_version,
        profile_label=layout.profile_label,
        blocks=layout.blocks,
        default_shuffle=layout.default_shuffle,
        shuffle_supported=layout.shuffle_supported,
    )


def _raw_answers(sheet: ScaleSheet, values: list[object]) -> str:
    return json.dumps(
        {
            "answers": [
                {"slot_id": slot.slot_id, "answer": value}
                for slot, value in zip(sheet.slots, values)
            ]
        },
        ensure_ascii=False,
    )


class TypedScaleSchemaTests(unittest.TestCase):
    def test_human_rights_requires_ten_episodes_and_builds_sixty_slots(self):
        scenarios = [
            (
                f"<Episode {episode}> Fixture episode {episode}\n"
                "I. Two possible viewpoints.\n"
                "1. Viewpoint one.\n"
                "2. Viewpoint two.\n"
                "II. Two possible outcomes.\n"
                "1. Outcome one.\n"
                "2. Outcome two.\n"
                "III. Two possible actions.\n"
                "1. Action one.\n"
                "2. Action two."
            )
            for episode in range(1, 11)
        ]
        layout = build_profile_layout(
            "human_rights_v1",
            ["Human Rights Sensitivity Scale", "Use 1–5.", *scenarios],
            "en",
            [],
        )
        self.assertEqual(len(layout.blocks), 10)
        self.assertEqual([len(block.slots) for block in layout.blocks], [6] * 10)
        slots = [slot for block in layout.blocks for slot in block.slots]
        self.assertEqual(len(slots), 60)
        self.assertEqual(slots[-1].slot_id, "human_rights_e10_action_02")

        with self.assertRaisesRegex(ProfileBuildError, "expected 10 episode cells"):
            build_profile_layout(
                "human_rights_v1",
                ["Human Rights Sensitivity Scale", "Use 1–5.", *scenarios[:6]],
                "en",
                [],
            )

    def test_registry_covers_all_declared_flat_scales(self):
        self.assertEqual(len(SCALE_NAME_PROFILES), 24)
        self.assertTrue(
            (set(SCALE_NAME_PROFILES.values()) - {"kohlberg_mji_v1"}).issubset(TYPED_FLAT_SPECS)
        )
        self.assertEqual(
            identify_profile("民主的通俗定义-评分表"),
            "popular_democracy_definition_v1",
        )

    def test_dilemma_profiles_are_explicitly_non_shuffleable(self):
        for profile_id in (
            "trolley_dilemma_v1",
            "footbridge_dilemma_v1",
            "prisoners_dilemma_v1",
        ):
            with self.subTest(profile_id=profile_id):
                spec = TYPED_FLAT_SPECS[profile_id]
                self.assertFalse(spec.default_shuffle)
                self.assertFalse(spec.shuffle_supported)
                self.assertEqual(spec.shuffle_policy, "fixed")

    def test_human_nature_uses_six_discrete_scores_without_zero(self):
        sheet = _typed_sheet("human_nature_philosophy_v1")
        plan = build_display_plan(sheet, False)
        valid = parse_profile_answers(
            _raw_answers(sheet, [3, 2, 1, -1, -2, -3] + [1] * 14), plan
        )
        self.assertEqual(valid.status, "ok")
        self.assertEqual(
            [answer.score for answer in valid.answers[:6]],
            [3.0, 2.0, 1.0, -1.0, -2.0, -3.0],
        )

        invalid = parse_profile_answers(
            _raw_answers(sheet, [0] + [1] * 19), plan
        )
        self.assertEqual(invalid.status, "partial")
        self.assertEqual(invalid.answers[0].parse_status, "invalid_choice")

    def test_attribution_probability_slots_allow_decimals(self):
        numbered = "\n".join(f"{index}. Item {index}" for index in range(1, 11))
        scenarios = [
            (
                f"Scenario {index}\n"
                f"A. Rate the statements.\n{numbered}\n"
                "B. Give a probability from 0% to 100%.\n"
                "C. Give a probability from 0% to 100%."
            )
            for index in range(1, 7)
        ]
        groupness = (
            "Select a 1 to 7 score.\n"
            + "\n".join(f"{index}. Group {index}" for index in range(1, 7))
        )
        layout = build_profile_layout(
            "attribution_bias_v1",
            ["Attribution Bias Task", "Use 1 to 7.", *scenarios, groupness],
            "en",
            [],
        )
        sheet = ScaleSheet(
            language="en",
            title="Attribution Bias Task",
            instruction=layout.instruction,
            questions=[slot.question for block in layout.blocks for slot in block.slots],
            profile_id=layout.profile_id,
            profile_version=layout.profile_version,
            profile_label=layout.profile_label,
            blocks=layout.blocks,
            default_shuffle=layout.default_shuffle,
            shuffle_supported=layout.shuffle_supported,
        )
        probability_slots = [
            slot for slot in sheet.slots if slot.metadata.get("subtask", "").endswith("probability")
        ]
        self.assertEqual(len(probability_slots), 12)
        self.assertTrue(all(slot.response_type == "number" for slot in probability_slots))

        plan = build_display_plan(sheet, False)
        values: list[object] = []
        for slot in plan.slots:
            values.append(50.5 if slot.response_type == "number" else 4)
        parsed = parse_profile_answers(_raw_answers(sheet, values), plan)
        self.assertEqual(parsed.status, "ok")
        self.assertTrue(
            all(answer.score == 50.5 for slot, answer in zip(plan.slots, parsed.answers) if slot.response_type == "number")
        )

    def test_integer_scale_rejects_fraction_and_out_of_range(self):
        sheet = _typed_sheet("popular_democracy_definition_v1")
        plan = build_display_plan(sheet, False)
        prompt = build_prompt(sheet, plan, language="ch")
        self.assertIn("General instruction.", prompt)
        self.assertIn("Fixture item 1.", prompt)
        self.assertNotIn("只能填 1–10", prompt)
        self.assertNotIn("民主的通俗定义量表", prompt)

        valid = parse_profile_answers(_raw_answers(sheet, [5] * 10), plan)
        self.assertEqual(valid.status, "ok")

        invalid_values: list[object] = [5] * 10
        invalid_values[0] = 11
        invalid_values[1] = 5.5
        invalid = parse_profile_answers(_raw_answers(sheet, invalid_values), plan)
        self.assertEqual(invalid.status, "partial")
        self.assertEqual(invalid.answers[0].parse_status, "above_maximum")
        self.assertEqual(invalid.answers[1].parse_status, "invalid_integer")
        self.assertEqual(invalid.answers[0].answer, 11)
        self.assertEqual(invalid.answers[1].answer, 5.5)

    def test_free_typed_prompt_omits_source_title_and_keeps_instruction(self):
        sheet = _typed_sheet("democracy_cognition_v1")
        prompt = build_prompt(
            sheet,
            build_display_plan(sheet, False),
            language="ch",
            contract="free",
        )
        self.assertFalse(prompt.startswith("Fixture\n"))
        self.assertIn("General instruction.", prompt)
        self.assertIn("1. Fixture item 1.", prompt)
        self.assertNotIn("民主感知量表（1–5）", prompt)
        self.assertNotIn("民主认知量表（1–5）", prompt)

    def test_percentage_scale_allows_decimal_but_enforces_bounds(self):
        sheet = _typed_sheet("change_expectation_v1")
        plan = build_display_plan(sheet, False)
        valid = parse_profile_answers(
            _raw_answers(sheet, [0, 12.5, "99.75", 100]), plan
        )
        self.assertEqual(valid.status, "ok")
        self.assertEqual([item.score for item in valid.answers], [0.0, 12.5, 99.75, 100.0])

        invalid = parse_profile_answers(
            _raw_answers(sheet, [-0.1, 50, 100.1, 75]), plan
        )
        self.assertEqual(invalid.status, "partial")
        self.assertEqual(invalid.answers[0].parse_status, "below_minimum")
        self.assertEqual(invalid.answers[2].parse_status, "above_maximum")

    def test_modern_racism_accepts_numeric_choices_and_x(self):
        sheet = _typed_sheet("modern_racism_v1")
        plan = build_display_plan(sheet, False)
        parsed = parse_profile_answers(
            _raw_answers(sheet, [2, "X", -1, 0, 1, -2, "2", "x"]), plan
        )
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(parsed.answers[0].answer, "2")
        self.assertEqual(parsed.answers[0].score, 2.0)
        self.assertEqual(parsed.answers[1].answer, "X")
        self.assertIsNone(parsed.answers[1].score)
        self.assertEqual(parsed.answers[7].answer, "X")

        invalid = parse_profile_answers(
            _raw_answers(sheet, [3, 0, 0, 0, 0, 0, 0, 0]), plan
        )
        self.assertEqual(invalid.status, "partial")
        self.assertEqual(invalid.answers[0].parse_status, "invalid_choice")

    def test_dilemmas_share_fixed_typed_slots(self):
        trolley = _typed_sheet("trolley_dilemma_v1")
        footbridge = _typed_sheet("footbridge_dilemma_v1")
        prisoner = _typed_sheet("prisoners_dilemma_v1")
        expected_common = [
            "final_choice",
            "moral_acceptability",
            "choice_confidence",
            "estimated_choice_a_count",
        ]
        self.assertEqual([slot.slot_id for slot in trolley.slots], expected_common)
        self.assertEqual([slot.slot_id for slot in footbridge.slots], expected_common)
        self.assertEqual([slot.slot_id for slot in prisoner.slots[:4]], expected_common)
        self.assertEqual(
            prisoner.slots[4].slot_id, "estimated_other_choice_a_probability"
        )
        self.assertFalse(trolley.shuffle_supported)
        self.assertFalse(build_display_plan(trolley, True).shuffle_enabled)

        invalid = parse_profile_answers(
            _raw_answers(trolley, ["C", 8, 5, 101]),
            build_display_plan(trolley, False),
        )
        self.assertEqual(invalid.status, "partial")
        self.assertEqual(
            [answer.parse_status for answer in invalid.answers],
            ["invalid_choice", "above_maximum", "parsed", "above_maximum"],
        )

    def test_ios_accepts_corrected_english_relatives_item(self):
        cells = [
            "Inclusion of Other in the Self Scale",
            "image formula",
            "The pictures symbolize relationships.",
            "1. One circle represents someone and the other represents colleagues.",
            "2. One circle represents someone and the other represents family members.",
            "3. One circle represents someone and the other represents relatives.",
            "4. One circle represents someone and the other represents friends.",
        ]
        layout = build_profile_layout(
            "ios_v1",
            cells,
            "en",
            [ScaleImage(anchor="A2", mime_type="image/png", data=b"png")],
        )
        self.assertEqual(len(layout.blocks[0].slots), 4)
        self.assertEqual(layout.blocks[0].slots[2].slot_id, "ios_relative")
        self.assertTrue(all(slot.response_type == "ios_pair" for slot in layout.blocks[0].slots))
        self.assertEqual(layout.blocks[0].slots[0].choices, ["1", "2", "3", "4", "5", "6", "7"])
        self.assertIsNone(layout.blocks[0].slots[2].normalization_note)
        self.assertTrue(layout.shuffle_supported)
        self.assertTrue(layout.default_shuffle)

        sheet = ScaleSheet(
            language="en",
            title=cells[0],
            instruction=layout.instruction,
            questions=[slot.question for block in layout.blocks for slot in block.slots],
            images=[ScaleImage(anchor="A2", mime_type="image/png", data=b"png")],
            profile_id=layout.profile_id,
            profile_version=layout.profile_version,
            profile_label=layout.profile_label,
            blocks=layout.blocks,
            default_shuffle=layout.default_shuffle,
            shuffle_supported=layout.shuffle_supported,
        )
        self.assertTrue(build_display_plan(sheet, True).shuffle_enabled)
        shuffled = build_display_plan(sheet, True, random.Random(2))
        self.assertEqual(
            {slot.slot_id for slot in shuffled.slots},
            {slot.slot_id for slot in build_display_plan(sheet, False).slots},
        )
        self.assertNotEqual(
            [slot.slot_id for slot in shuffled.slots],
            [slot.slot_id for slot in build_display_plan(sheet, False).slots],
        )
        self.assertTrue(
            all(slot.choices == ["1", "2", "3", "4", "5", "6", "7"] for slot in shuffled.slots)
        )
        prompt = build_prompt(sheet, build_display_plan(sheet, False), language="en")
        self.assertIn("The pictures symbolize relationships.", prompt)
        self.assertIn("colleagues", prompt)
        self.assertNotIn("pair-of-circles diagram", prompt)
        self.assertNotIn("two-circle code", prompt)

        free = parse_free_answers(
            "1. Pair (3)\n2. 图示5\n3. 图 **(2)**\n4. 7",
            build_display_plan(sheet, False).slots,
        )
        self.assertEqual(free.status, "ok")
        self.assertEqual([answer.answer for answer in free.answers], ["3", "5", "2", "7"])
        self.assertEqual([answer.score for answer in free.answers], [3, 5, 2, 7])

        ios_short_wrapper = parse_free_answers(
            "1. 选（1）",
            build_display_plan(sheet, False).slots[:1],
        )
        self.assertEqual(ios_short_wrapper.status, "ok")
        self.assertEqual(ios_short_wrapper.answers[0].answer, "1")

        table = parse_free_answers(
            "| item | answer |\n|---|---|\n| 1 | Pair (4) |\n| 2 | 图示(1) |\n| 3 | 2 |\n| 4 | **6** |",
            build_display_plan(sheet, False).slots,
        )
        self.assertEqual(table.status, "ok")
        self.assertEqual([answer.answer for answer in table.answers], ["4", "1", "2", "6"])
        self.assertEqual([answer.score for answer in table.answers], [4, 1, 2, 6])

        strict = parse_profile_answers(
            json.dumps(
                {
                    "answers": [
                        {"slot_id": slot.slot_id, "answer": code}
                        for slot, code in zip(
                            build_display_plan(sheet, False).slots,
                            ["3", "5", "2", "7"],
                        )
                    ]
                },
                ensure_ascii=False,
            ),
            build_display_plan(sheet, False),
        )
        self.assertEqual(strict.status, "ok")
        self.assertEqual([answer.score for answer in strict.answers], [3, 5, 2, 7])

        strict_wrapped = parse_profile_answers(
            json.dumps(
                {
                    "answers": [
                        {"slot_id": slot.slot_id, "answer": value}
                        for slot, value in zip(
                            build_display_plan(sheet, False).slots,
                            ["Pair (5)", "图示(2)", "7", "3-4"],
                        )
                    ]
                },
                ensure_ascii=False,
            ),
            build_display_plan(sheet, False),
        )
        self.assertEqual(strict_wrapped.status, "partial")
        self.assertEqual([answer.answer for answer in strict_wrapped.answers[:3]], ["5", "2", "7"])
        self.assertEqual(strict_wrapped.answers[3].answer, "3-4")
        self.assertEqual(strict_wrapped.answers[3].parse_status, "invalid_ios_pair")

        ambiguous = parse_free_answers(
            "1. Pair (3) or (5)\n2. 2\n3. 3\n4. 4",
            build_display_plan(sheet, False).slots,
        )
        self.assertEqual(ambiguous.answers[0].answer, "Pair (3) or (5)")
        self.assertIsNone(ambiguous.answers[0].score)
        self.assertEqual(ambiguous.answers[0].parse_status, "unparsed")

    def test_intuitive_reasoning_prompt_keeps_source_instructions_without_backend_labels(self):
        rating_instruction = (
            "Based on general conditions, please explicitly give one score for each option: "
            "-3=definitely false, -2=probably false, -1=slightly false, "
            "0=neither/not familiar, +1=slightly true, +2=probably true, +3=definitely true."
        )
        cells = [
            "Intuitive (vs. Formal) Reasoning Task",
            "For each problem, decide whether the conclusion follows logically from the premises.",
            *[f"{number}. Premises and conclusion for argument {number}." for number in range(1, 25)],
            rating_instruction,
            *[f"{number}. Conclusion rating item {number}." for number in range(1, 17)],
        ]
        layout = build_profile_layout("intuitive_reasoning_v1", cells, "en", [])
        sheet = ScaleSheet(
            language="en",
            title=cells[0],
            instruction=layout.instruction,
            questions=[slot.question for block in layout.blocks for slot in block.slots],
            profile_id=layout.profile_id,
            profile_version=layout.profile_version,
            profile_label=layout.profile_label,
            blocks=layout.blocks,
        )

        prompt = build_prompt(sheet, build_display_plan(sheet, False), language="en")
        self.assertIn(cells[1], prompt)
        self.assertIn(cells[2], prompt)
        self.assertIn(cells[-1], prompt)
        self.assertEqual(prompt.count(rating_instruction), 1)
        self.assertNotIn("Logical validity (concrete arguments)", prompt)
        self.assertNotIn("Logical validity (abstract control arguments)", prompt)
        self.assertNotIn("Conclusion believability ratings", prompt)


if __name__ == "__main__":
    unittest.main()
