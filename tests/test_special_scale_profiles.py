from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.prompting import (  # noqa: E402
    FREE_PARSER_VERSION,
    build_display_plan,
    build_prompt,
    build_result_payload,
    parse_kohlberg_free_answers,
    parse_free_answers,
    parse_profile_answers,
    remap_to_original,
)
from app.runner import BatchRunner, RunnerConfig, ScaleRunConfig  # noqa: E402
from app.scale_loader import ScaleFile, ScaleSheet, ScaleSlot, ScaleTaskBlock  # noqa: E402
from app.scale_profiles import (  # noqa: E402
    build_profile_layout,
    identify_profile,
    select_kohlberg_form,
)


def _profile_sheet() -> ScaleSheet:
    scenario_1 = ScaleTaskBlock(
        block_id="scenario_1",
        label="Scenario 1",
        context="Context one must travel with its answers.",
        slots=[
            ScaleSlot(
                slot_id="s1_choice",
                question="Does the conclusion follow?",
                response_type="choice",
                choices=["YES", "NO"],
                context_id="scenario_1",
            ),
            ScaleSlot(
                slot_id="s1_rating",
                question="Rate it.",
                response_type="integer",
                minimum=1,
                maximum=7,
                context_id="scenario_1",
            ),
        ],
        shuffle_policy="block",
    )
    scenario_2 = ScaleTaskBlock(
        block_id="scenario_2",
        label="Scenario 2",
        context="Context two must travel with its answer.",
        slots=[
            ScaleSlot(
                slot_id="s2_rating",
                question="Give a percentage.",
                response_type="integer",
                minimum=0,
                maximum=100,
                context_id="scenario_2",
            )
        ],
        shuffle_policy="block",
    )
    fixed = ScaleTaskBlock(
        block_id="fixed",
        label="Fixed follow-up",
        slots=[
            ScaleSlot(
                slot_id="fixed_rating",
                question="Final rating.",
                response_type="integer",
                minimum=-3,
                maximum=3,
                linked_slot_id="s1_choice",
            )
        ],
        shuffle_policy="fixed",
    )
    return ScaleSheet(
        language="ch",
        title="Special profile fixture",
        instruction="General instruction.",
        questions=[slot.question for block in (scenario_1, scenario_2, fixed) for slot in block.slots],
        profile_id="fixture_v1",
        profile_version="1",
        profile_label="Fixture profile",
        blocks=[scenario_1, scenario_2, fixed],
        default_shuffle=False,
        shuffle_supported=True,
    )


def _attribution_sheet() -> ScaleSheet:
    blocks = []
    for scenario in range(1, 7):
        slots = [
            ScaleSlot(
                slot_id=f"attribution_s{scenario:02d}_a{item:02d}",
                question=f"Scenario {scenario} A item {item}.",
                response_type="integer",
                minimum=1,
                maximum=7,
                context_id=f"attribution_s{scenario:02d}",
                section_id="attribution_agreement",
                section_label=f"A. The statements for scenario {scenario}.",
                metadata={"scenario": scenario, "subtask": "attribution_rating"},
            )
            for item in range(1, 11)
        ]
        slots.extend(
            [
                ScaleSlot(
                    slot_id=f"attribution_s{scenario:02d}_b",
                    question=f"Another person probability for scenario {scenario}.",
                    response_type="number",
                    minimum=0,
                    maximum=100,
                    context_id=f"attribution_s{scenario:02d}",
                    section_id="other_actor_probability",
                    metadata={"scenario": scenario, "subtask": "other_actor_probability"},
                ),
                ScaleSlot(
                    slot_id=f"attribution_s{scenario:02d}_c",
                    question=f"Same person probability for scenario {scenario}.",
                    response_type="number",
                    minimum=0,
                    maximum=100,
                    context_id=f"attribution_s{scenario:02d}",
                    section_id="same_actor_probability",
                    metadata={"scenario": scenario, "subtask": "same_actor_probability"},
                ),
            ]
        )
        blocks.append(
            ScaleTaskBlock(
                block_id=f"attribution_s{scenario:02d}",
                label=f"Scenario {scenario}",
                context=f"Situation {scenario}: scenario context.",
                slots=slots,
                shuffle_policy="block",
            )
        )

    groupness_slots = [
        ScaleSlot(
            slot_id=f"attribution_groupness_g{item:02d}",
            question=f"Groupness item {item}.",
            response_type="integer",
            minimum=1,
            maximum=7,
            context_id="attribution_groupness",
            section_id="groupness",
            metadata={"subtask": "groupness"},
        )
        for item in range(1, 7)
    ]
    blocks.append(
        ScaleTaskBlock(
            block_id="attribution_groupness",
            label="Groupness ratings",
            instruction="Groupness instruction.",
            slots=groupness_slots,
            shuffle_policy="fixed",
        )
    )
    return ScaleSheet(
        language="en",
        title="Attribution Bias Task",
        instruction="Based on general conditions, give one score.",
        questions=[slot.question for block in blocks for slot in block.slots],
        profile_id="attribution_bias_v1",
        profile_version="1",
        profile_label="Attribution bias task",
        blocks=blocks,
        default_shuffle=True,
        shuffle_supported=True,
    )


class SpecialProfilePipelineTests(unittest.TestCase):
    def test_kohlberg_story_subscale_is_one_open_ended_slot(self):
        profile_id = identify_profile(
            "科尔伯格-偷药困境-纯原版-评分表",
            "科尔伯格——偷药困境——纯原版",
        )
        self.assertEqual(profile_id, "kohlberg_story_v1")
        layout = build_profile_layout(
            profile_id,
            [
                "科尔伯格——偷药困境——纯原版",
                "请根据下面的情境，回答后续问题。",
                "情境全文。\n1. 第一个问题？\n2. 第二个问题？",
            ],
            "ch",
            [],
        )
        self.assertEqual(len(layout.blocks), 1)
        self.assertEqual(len(layout.blocks[0].slots), 1)
        self.assertTrue(layout.blocks[0].slots[0].metadata["open_ended"])
        self.assertFalse(layout.shuffle_supported)

        sheet = ScaleSheet(
            language="ch",
            title="科尔伯格——偷药困境——纯原版",
            instruction=layout.instruction,
            questions=[layout.blocks[0].slots[0].question],
            profile_id=layout.profile_id,
            profile_version=layout.profile_version,
            profile_label=layout.profile_label,
            blocks=layout.blocks,
            default_shuffle=layout.default_shuffle,
            shuffle_supported=layout.shuffle_supported,
        )
        prompt = build_prompt(
            sheet,
            build_display_plan(sheet, False),
            language="ch",
            contract="free",
        )
        self.assertIn("情境全文。", prompt)
        self.assertIn("1. 第一个问题？", prompt)
        parsed = parse_free_answers(
            "1. 第一个问题的回答。\n2. 第二个问题的回答。",
            sheet.slots,
        )
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(parsed.answers[0].parse_status, "text_response")
        self.assertIn("第二个问题的回答", parsed.answers[0].answer)

    def test_kohlberg_profile_excludes_scoring_note_and_preserves_forms(self):
        cells = [
            "Moral Judgment Interview (MJI)",
            "Read each dilemma aloud and record the open-ended response.",
            "Story A1",
            "Story A2",
            "Story A3",
            "Story B1",
            "Story B2",
            "Story B3",
            "Story C1",
            "Story C2",
            "Story C3",
            "Scoring note: do not sum item ratings.",
        ]
        self.assertEqual(identify_profile("科尔伯格道德判断访谈-普适版-评分表", cells[0]), "kohlberg_mji_v1")
        layout = build_profile_layout("kohlberg_mji_v1", cells, "en", [])
        self.assertEqual(len(layout.blocks), 9)
        self.assertEqual(sum(len(block.slots) for block in layout.blocks), 9)
        self.assertFalse(layout.default_shuffle)
        self.assertFalse(layout.shuffle_supported)
        self.assertNotIn("Scoring note", "\n".join(slot.question for block in layout.blocks for slot in block.slots))

        sheet = ScaleSheet(
            language="en",
            title=cells[0],
            instruction=layout.instruction,
            questions=[slot.question for block in layout.blocks for slot in block.slots],
            profile_id=layout.profile_id,
            profile_version=layout.profile_version,
            profile_label=layout.profile_label,
            blocks=layout.blocks,
            default_shuffle=layout.default_shuffle,
            shuffle_supported=layout.shuffle_supported,
        )
        free_prompt = build_prompt(
            sheet,
            build_display_plan(sheet, False),
            language="en",
            contract="free",
        )
        self.assertNotIn(cells[0], free_prompt)
        self.assertIn(cells[1], free_prompt)
        self.assertIn("Story A1", free_prompt)
        self.assertNotIn("Scoring note", free_prompt)

        form_a = select_kohlberg_form(sheet, "A")
        self.assertEqual(len(form_a.blocks), 3)
        self.assertEqual([block.metadata["dilemma"] for block in form_a.blocks], ["III", "III'", "I"])
        self.assertEqual(form_a.profile_version, "1-form-A")
        self.assertFalse(form_a.shuffle_supported)
        form_a_prompt = build_prompt(
            form_a,
            build_display_plan(form_a, True),
            language="en",
            contract="free",
        )
        self.assertEqual(form_a_prompt.count("Story A1"), 1)
        self.assertEqual(form_a_prompt.count("Story A2"), 1)
        self.assertEqual(form_a_prompt.count("Story A3"), 1)
        self.assertNotIn("Story B1", form_a_prompt)

    def test_kohlberg_decoder_uses_story_labels_not_nested_question_numbers(self):
        slots = [
            ScaleSlot(
                slot_id="story1",
                question="故事一",
                response_type="text",
                metadata={"open_ended": True},
            ),
            ScaleSlot(
                slot_id="story2",
                question="故事二",
                response_type="text",
                metadata={"open_ended": True},
            ),
        ]
        raw = (
            "Story 1:\n总体应当救五个人。\n1. 理由一\n2. 理由二\n"
            "Story 2:\n应当尊重规则。\n1. 追问回答"
        )
        parsed = parse_kohlberg_free_answers(raw, slots)
        self.assertEqual(parsed.status, "ok")
        self.assertIn("救五个人", parsed.answers[0].answer)
        self.assertNotIn("尊重规则", parsed.answers[0].answer)
        self.assertIn("尊重规则", parsed.answers[1].answer)

        unsafe = parse_kohlberg_free_answers(
            "1. 故事一回答\n1. 故事一追问\n2. 故事二回答", slots
        )
        self.assertEqual(unsafe.status, "unparsed")
        self.assertTrue(all(answer.answer is None for answer in unsafe.answers))

    def test_runner_applies_kohlberg_form_before_result_and_plan(self):
        cells = [
            "Moral Judgment Interview (MJI)",
            "Read each dilemma aloud and record the open-ended response.",
            "Story A1", "Story A2", "Story A3",
            "Story B1", "Story B2", "Story B3",
            "Story C1", "Story C2", "Story C3",
            "Scoring note: do not sum item ratings.",
        ]
        layout = build_profile_layout("kohlberg_mji_v1", cells, "en", [])
        sheet = ScaleSheet(
            language="en",
            title=cells[0],
            instruction=layout.instruction,
            questions=[slot.question for block in layout.blocks for slot in block.slots],
            profile_id=layout.profile_id,
            profile_version=layout.profile_version,
            profile_label=layout.profile_label,
            blocks=layout.blocks,
            default_shuffle=layout.default_shuffle,
            shuffle_supported=layout.shuffle_supported,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scale = ScaleFile(path=root / "fixture.xlsx", name="科尔伯格", en=sheet)
            config = RunnerConfig(
                results_root=root / "results",
                logs_root=root / "logs",
                max_retries=1,
                prompt_contract="free",
                scale_cfgs={
                    "科尔伯格": ScaleRunConfig(
                        scale_name="科尔伯格", mji_form="A"
                    )
                },
            )
            runner = BatchRunner([scale], config)
            effective = runner._effective_sheet(scale, sheet)
            self.assertEqual(effective.n_items, 3)
            self.assertEqual(effective.profile_version, "1-form-A")
            runner._write_trial_plan(
                scale=scale,
                sheet=effective,
                language="en",
                provider="deepseek",
                model="deepseek-v4-pro",
                order_ids=[1],
            )
            plan_record = json.loads(
                (root / "results" / "adhoc" / "plan.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(plan_record["n_items"], 3)
            self.assertEqual(plan_record["profile_version"], "1-form-A")
            with patch(
                "app.runner.call_api",
                return_value="1. 结合故事作答。\n2. 结合故事作答。\n3. 结合故事作答。",
            ):
                result_path = runner._one_call(
                    scale, sheet, "en", 1, "deepseek", "deepseek-v4-pro"
                )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["item_count"], 3)
        self.assertEqual(payload["scale_profile"]["version"], "1-form-A")
        self.assertNotIn("Story B1", payload["prompt"])

    def test_display_plan_moves_whole_blocks_but_keeps_fixed_followup(self):
        sheet = _profile_sheet()
        plan = build_display_plan(sheet, True, random.Random(7))
        self.assertEqual(plan.blocks[-1].block_id, "fixed")
        self.assertEqual(
            {block.block_id for block in plan.blocks[:2]}, {"scenario_1", "scenario_2"}
        )
        self.assertEqual(len(plan.slots), 4)
        # Every slot remains attached to exactly one prompt block.
        self.assertEqual(
            [slot.slot_id for slot in plan.slots],
            [slot.slot_id for block in plan.blocks for slot in block.slots],
        )

    def test_attribution_shuffle_moves_six_complete_scenarios_only(self):
        sheet = _attribution_sheet()
        self.assertTrue(sheet.default_shuffle)
        plan = build_display_plan(sheet, True, random.Random(7))

        self.assertEqual(plan.blocks[-1].block_id, "attribution_groupness")
        self.assertEqual(
            {block.block_id for block in plan.blocks[:6]},
            {f"attribution_s{scenario:02d}" for scenario in range(1, 7)},
        )
        for block in plan.blocks[:6]:
            self.assertEqual(len(block.slots), 12)
            self.assertEqual(
                [slot.metadata["subtask"] for slot in block.slots],
                ["attribution_rating"] * 10
                + ["other_actor_probability", "same_actor_probability"],
            )
        self.assertEqual(len(plan.blocks[-1].slots), 6)

    def test_attribution_free_prompt_omits_title_and_preserves_followup_sections(self):
        sheet = _attribution_sheet()
        plan = build_display_plan(sheet, True, random.Random(7))
        prompt = build_prompt(sheet, plan, language="en", contract="free")

        self.assertNotIn("Attribution Bias Task", prompt)
        self.assertTrue(prompt.startswith("Based on general conditions, give one score."))
        self.assertEqual(prompt.count("Groupness instruction."), 1)
        self.assertEqual(prompt.count("A. The statements for scenario"), 6)
        self.assertEqual(prompt.count("B. Another person probability for scenario"), 6)
        self.assertEqual(prompt.count("C. Same person probability for scenario"), 6)
        self.assertNotIn("slot_id", prompt)
        self.assertNotIn("Response slot", prompt)

    def test_profile_prompt_and_parser_use_stable_ids_and_validate_ranges(self):
        sheet = _profile_sheet()
        plan = build_display_plan(sheet, False)
        prompt = build_prompt(sheet, plan, language="ch")
        self.assertNotIn("s1_choice", prompt)
        self.assertNotIn('"slot_id"', prompt)
        self.assertIn("General instruction.", prompt)
        self.assertIn("Context one must travel with its answers.", prompt)
        self.assertNotIn("只能填 1–7", prompt)
        self.assertNotIn("Special profile fixture", prompt)
        self.assertNotIn("Scenario 1", prompt)
        self.assertNotIn("Scenario 2", prompt)
        self.assertNotIn("Fixed follow-up", prompt)
        self.assertNotIn("任务组", prompt)

        raw = json.dumps(
            {
                "answers": [
                    {"slot_id": "s1_choice", "answer": "是"},
                    {"slot_id": "s1_rating", "answer": 7},
                    {"slot_id": "s2_rating", "answer": 40},
                    {"slot_id": "fixed_rating", "answer": -3},
                ]
            },
            ensure_ascii=False,
        )
        parsed = parse_profile_answers(raw, plan)
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(parsed.answers[0].answer, "YES")
        self.assertEqual(parsed.answers[1].score, 7.0)

        invalid = json.dumps(
            {
                "answers": [
                    {"slot_id": "s1_choice", "answer": "because YES is logical"},
                    {"slot_id": "s1_rating", "answer": 8},
                    {"slot_id": "s2_rating", "answer": 40},
                    {"slot_id": "fixed_rating", "answer": -3},
                ]
            }
        )
        invalid_result = parse_profile_answers(invalid, plan)
        self.assertEqual(invalid_result.status, "partial")
        self.assertEqual(invalid_result.answers[0].parse_status, "invalid_choice")
        self.assertEqual(invalid_result.answers[1].parse_status, "above_maximum")
        self.assertEqual(invalid_result.answers[0].answer, "because YES is logical")
        self.assertEqual(invalid_result.answers[1].answer, 8)

    def test_profile_parser_recovers_exact_slot_jsonl(self):
        sheet = _profile_sheet()
        plan = build_display_plan(sheet, False)
        raw = "\n".join(
            [
                '{"slot_id":"s1_choice","answer":"NO"}',
                '{"slot_id":"s1_rating","answer":2}',
                '{"slot_id":"s2_rating","answer":40}',
                '{"slot_id":"fixed_rating","answer":1}',
            ]
        )
        parsed = parse_profile_answers(raw, plan)
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(parsed.recovery, "slot_jsonl_v1")
        self.assertEqual([answer.answer for answer in parsed.answers], ["NO", 2, 40, 1])

    def test_profile_payload_keeps_canonical_ids_and_display_mapping(self):
        sheet = _profile_sheet()
        plan = build_display_plan(sheet, False)
        raw = (
            '{"answers":[{"slot_id":"s1_choice","answer":"NO"},'
            '{"slot_id":"s1_rating","answer":2},'
            '{"slot_id":"s2_rating","answer":99},'
            '{"slot_id":"fixed_rating","answer":1}]}'
        )
        parsed = parse_profile_answers(raw, plan)
        original = remap_to_original(parsed.answers, plan.original_indices)
        payload = build_result_payload(
            scale_name="fixture",
            language="ch",
            provider="deepseek",
            model="deepseek-v4-pro",
            temperature=0.0,
            order_id=1,
            seed=None,
            order=plan.original_indices,
            shuffle_enabled=False,
            sheet=sheet,
            answers_original=original,
            parse_result=parsed,
            raw_response=raw,
            prompt=build_prompt(sheet, plan, language="ch"),
            display_plan=plan,
        )
        self.assertEqual(payload["result_schema_version"], "scale-result-v3")
        self.assertEqual(payload["items"][0]["slot_id"], "s1_choice")
        self.assertEqual(payload["blocks"][0]["block_id"], "scenario_1")
        self.assertEqual(
            payload["display_plan"]["blocks"][0]["slots"][0]["slot_id"], "s1_choice"
        )

    def test_runner_uses_profile_parser_and_writes_v3_without_api_network(self):
        sheet = _profile_sheet()
        raw = (
            '{"answers":[{"slot_id":"s1_choice","answer":"NO"},'
            '{"slot_id":"s1_rating","answer":2},'
            '{"slot_id":"s2_rating","answer":99},'
            '{"slot_id":"fixed_rating","answer":1}]}'
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scale = ScaleFile(path=root / "fixture.xlsx", name="fixture", ch=sheet)
            config = RunnerConfig(
                results_root=root / "results", logs_root=root / "logs", max_retries=1
            )
            # Explicit shuffle is accepted, but the profile planner preserves
            # block integrity and records its mapping rather than flat-shuffling.
            config.scale_cfgs[scale.name] = ScaleRunConfig(
                scale_name=scale.name, shuffle_items=True
            )
            runner = BatchRunner([scale], config)
            with patch("app.runner.call_api", return_value=raw) as mocked_call:
                result_path = runner._one_call(
                    scale, sheet, "ch", 1, "deepseek", "deepseek-v4-pro"
                )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["result_schema_version"], "scale-result-v3")
        self.assertEqual(payload["response_parser"]["version"], FREE_PARSER_VERSION)
        self.assertEqual(payload["model_identity"]["provider"], "deepseek")
        self.assertEqual(payload["model_identity"]["requested_model"], "deepseek-v4-pro")
        self.assertIn("recorded_at_utc", payload["model_identity"])
        self.assertIn("recorded_at_local", payload["model_identity"])
        self.assertIn("request_started_at_utc", payload["model_identity"])
        self.assertIn("request_started_at_local", payload["model_identity"])
        self.assertIn("request_completed_at_utc", payload["model_identity"])
        self.assertIn("request_completed_at_local", payload["model_identity"])
        self.assertEqual(len(payload["condition_hash"]), 64)
        self.assertEqual(len(payload["execution_hash"]), 64)
        self.assertIn("display_plan", payload)
        self.assertEqual(payload["items"][0]["slot_id"], "s1_choice")
        self.assertEqual(mocked_call.call_count, 1)

    def test_runner_saves_unparsed_response_as_trial_result(self):
        sheet = _profile_sheet()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scale = ScaleFile(path=root / "fixture.xlsx", name="fixture", ch=sheet)
            config = RunnerConfig(
                results_root=root / "results",
                logs_root=root / "logs",
                max_retries=1,
                retry_delay=0,
            )
            runner = BatchRunner([scale], config)
            with patch("app.runner.call_api", return_value="not json"):
                result_path = runner._one_call(
                    scale, sheet, "ch", 1, "deepseek", "deepseek-v4-pro"
                )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            attempts = list((root / "results").rglob("trial_*_attempt_*.json"))
            attempt = json.loads(attempts[0].read_text(encoding="utf-8"))
        self.assertEqual(payload["response_status"], "unparsed")
        self.assertEqual(payload["response_parser"]["status"], "unparsed")
        self.assertEqual(payload["raw_response"], "not json")
        self.assertEqual(payload["item_count"], 4)
        self.assertEqual(payload["read_item_count"], 0)
        self.assertFalse(payload["all_items_read"])
        self.assertEqual(attempt["status"], "unparsed")

    def test_runner_saves_refusal_as_trial_result_without_retry(self):
        sheet = _profile_sheet()
        refusal = "抱歉，我无法对这些对象进行评价。"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scale = ScaleFile(path=root / "fixture.xlsx", name="fixture", ch=sheet)
            config = RunnerConfig(
                results_root=root / "results", logs_root=root / "logs", max_retries=3
            )
            runner = BatchRunner([scale], config)
            with patch("app.runner.call_api", return_value=refusal) as mocked_call:
                result_path = runner._one_call(
                    scale, sheet, "ch", 1, "deepseek", "deepseek-v4-pro"
                )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["response_status"], "refusal")
        self.assertEqual(payload["raw_response"], refusal)
        self.assertEqual(payload["read_item_count"], 0)
        self.assertEqual(mocked_call.call_count, 1)

    def test_runner_saves_partial_parse_as_trial_result(self):
        sheet = _profile_sheet()
        # The response contains valid answers for three slots and omits one.
        # It must be retained as a partial result rather than discarded as a
        # parser failure.
        raw = "1. NO\n2. 2\n3. 40"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scale = ScaleFile(path=root / "fixture.xlsx", name="fixture", ch=sheet)
            config = RunnerConfig(
                results_root=root / "results", logs_root=root / "logs", max_retries=1
            )
            runner = BatchRunner([scale], config)
            with patch("app.runner.call_api", return_value=raw):
                result_path = runner._one_call(
                    scale, sheet, "ch", 1, "deepseek", "deepseek-v4-pro"
                )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            attempts = list((root / "results").rglob("trial_*_attempt_*.json"))
            attempt = json.loads(attempts[0].read_text(encoding="utf-8"))

        self.assertEqual(payload["response_parser"]["status"], "partial")
        self.assertEqual(payload["items"][0]["answer"], "NO")
        self.assertIsNone(payload["items"][3]["answer"])
        self.assertEqual(payload["items"][3]["parse_status"], "missing")
        self.assertEqual(attempt["status"], "partial")
        self.assertTrue(result_path.name.startswith("order_"))

    def test_runner_normalizes_legacy_local_timestamps(self):
        sheet = _profile_sheet()
        raw = (
            '{"answers":[{"slot_id":"s1_choice","answer":"NO"},'
            '{"slot_id":"s1_rating","answer":2},'
            '{"slot_id":"s2_rating","answer":99},'
            '{"slot_id":"fixed_rating","answer":1}]}'
        )
        legacy_metadata = {
            "provider": "deepseek",
            "requested_model": "deepseek-v4-pro",
            "recorded_at_utc": "2026-09-10T19:00:00+08:00",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scale = ScaleFile(path=root / "fixture.xlsx", name="fixture", ch=sheet)
            config = RunnerConfig(
                results_root=root / "results", logs_root=root / "logs", max_retries=1
            )
            runner = BatchRunner([scale], config)
            with patch(
                "app.runner.call_api",
                return_value={"text": raw, "model_metadata": legacy_metadata},
            ):
                result_path = runner._one_call(
                    scale, sheet, "ch", 1, "deepseek", "deepseek-v4-pro"
                )
            payload = json.loads(result_path.read_text(encoding="utf-8"))

        identity = payload["model_identity"]
        self.assertTrue(identity["recorded_at_utc"].endswith("+00:00"))
        self.assertTrue(identity["recorded_at_local"].endswith("+08:00"))
        self.assertTrue(identity["request_started_at_utc"].endswith("+00:00"))
        self.assertTrue(identity["request_started_at_local"].endswith("+08:00"))
        self.assertTrue(identity["request_completed_at_utc"].endswith("+00:00"))
        self.assertTrue(identity["request_completed_at_local"].endswith("+08:00"))
        self.assertEqual(identity["recorded_at_local"], "2026-09-10T19:00:00+08:00")


if __name__ == "__main__":
    unittest.main()
