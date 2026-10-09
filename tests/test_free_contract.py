from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.condition_design import build_condition_descriptor
from app.experiment_presets import (
    apply_preset,
    list_presets,
    parameter_guide,
    validate_preset_definition,
)
from app.generation_config import GenerationConfig
from app.prompting import (
    PROMPT_CONTRACTS,
    FREE_PARSER_VERSION,
    build_result_payload,
    build_prompt,
    classify_model_response,
    normalize_prompt_contract,
    parse_free_answers,
)
from app.runner import BatchRunner, RunnerConfig
from app.scale_loader import ScaleFile, ScaleSheet, ScaleSlot, ScaleTaskBlock


def flat_sheet() -> ScaleSheet:
    """A minimal un-profiled sheet whose questions already carry numbers."""
    return ScaleSheet(
        language="ch",
        title="测试量表",
        instruction="请为每题给出 1 到 7 的分数。",
        questions=[
            "1. 我不会为了团队牺牲个人利益。",
            "2. 我认为没有必要按团队意愿行事。",
            "3. 我维护团队的和谐。",
        ],
    )


def profiled_sheet() -> ScaleSheet:
    """A profile-backed sheet whose slots carry real answer types."""
    slots = [
        ScaleSlot(slot_id="q1", question="请评价第一条。", response_type="integer",
                  minimum=1, maximum=7),
        ScaleSlot(slot_id="q2", question="请评价第二条。", response_type="integer",
                  minimum=1, maximum=7),
        ScaleSlot(slot_id="q3", question="请评价第三条。", response_type="integer",
                  minimum=1, maximum=7),
    ]
    block = ScaleTaskBlock(
        block_id="b1", label="组块一", context="情境：某人做了一件事。",
        instruction="", slots=slots, shuffle_policy="fixed",
    )
    return ScaleSheet(
        language="ch",
        title="测试专属量表",
        instruction="请阅读情境后作答。",
        questions=[slot.question for slot in slots],
        profile_id="test_profile_v1",
        profile_label="测试专属",
        blocks=[block],
        shuffle_supported=False,
    )


def profiled_sheet_with_source_sections() -> ScaleSheet:
    """A profiled sheet whose section instructions are source content."""
    slots = [
        ScaleSlot(
            slot_id="q1", question="观点一。", response_type="integer",
            minimum=1, maximum=5, section_id="viewpoint",
            section_label="I. 你认为每种观点有多重要？",
        ),
        ScaleSlot(
            slot_id="q2", question="观点二。", response_type="integer",
            minimum=1, maximum=5, section_id="viewpoint",
            section_label="I. 你认为每种观点有多重要？",
        ),
        ScaleSlot(
            slot_id="q3", question="结果一。", response_type="integer",
            minimum=1, maximum=5, section_id="outcome",
            section_label="II. 你认为每种结果有多重要？",
        ),
        ScaleSlot(
            slot_id="q4", question="结果二。", response_type="integer",
            minimum=1, maximum=5, section_id="outcome",
            section_label="II. 你认为每种结果有多重要？",
        ),
        ScaleSlot(
            slot_id="q5", question="做法一。", response_type="integer",
            minimum=1, maximum=5, section_id="action",
            section_label="III. 如果你处在这个情境中，你会怎么做？",
        ),
        ScaleSlot(
            slot_id="q6", question="做法二。", response_type="integer",
            minimum=1, maximum=5, section_id="action",
            section_label="III. 如果你处在这个情境中，你会怎么做？",
        ),
    ]
    block = ScaleTaskBlock(
        block_id="b1", label="组块一", context="情境：某人做了一件事。",
        instruction="", slots=slots, shuffle_policy="fixed",
    )
    return ScaleSheet(
        language="ch", title="测试专属量表",
        instruction="请为每个选项评分。",
        questions=[slot.question for slot in slots],
        profile_id="test_profile_v1", profile_label="测试专属",
        blocks=[block], shuffle_supported=False,
    )


class FreePromptContractTests(unittest.TestCase):
    def test_contract_normalisation_rejects_unknown_values(self):
        self.assertEqual(normalize_prompt_contract(None), "unconstrained")
        self.assertEqual(normalize_prompt_contract("free"), "unconstrained")
        self.assertEqual(
            normalize_prompt_contract("free_scores_only"), "free_scores_only"
        )
        self.assertIn("free", PROMPT_CONTRACTS)
        with self.assertRaises(ValueError):
            normalize_prompt_contract("loose")

    def test_free_prompt_drops_every_answer_constraint(self):
        sheet = flat_sheet()
        prompt = build_prompt(sheet, [0, 1, 2], language="ch", contract="free")
        self.assertNotIn("作答槽位", prompt)
        self.assertNotIn("答案类型", prompt)
        self.assertNotIn("输出格式", prompt)
        self.assertNotIn("JSON", prompt)
        self.assertNotIn("测试量表", prompt)
        self.assertIn("请为每题给出 1 到 7 的分数。", prompt)

    def test_free_prompt_does_not_double_the_item_numbering(self):
        sheet = flat_sheet()
        prompt = build_prompt(sheet, [0, 1, 2], language="ch", contract="free")
        self.assertIn("1. 我不会为了团队牺牲个人利益。", prompt)
        self.assertNotIn("1. 1.", prompt)
        self.assertIn("3. 我维护团队的和谐。", prompt)

    def test_scores_only_prompt_adds_only_the_extra_no_explanation_instruction(self):
        sheet = flat_sheet()
        prompt = build_prompt(
            sheet, [0, 1, 2], language="ch", contract="free_scores_only"
        )
        self.assertIn("请仅给出题目要求的分数，不需要解释。", prompt)
        self.assertNotIn("作答槽位", prompt)
        self.assertNotIn("JSON", prompt)

    def test_default_and_legacy_prompts_never_add_answer_constraints(self):
        for contract in (None, "strict", "free", "unconstrained"):
            for sheet in (flat_sheet(), profiled_sheet()):
                with self.subTest(contract=contract, profile=sheet.profile_id):
                    prompt = build_prompt(sheet, list(range(sheet.n_items)), "ch", contract)
                    for generated in ("输出格式", "作答槽位", "JSON", "答案类型"):
                        self.assertNotIn(generated, prompt)
                    self.assertIn(sheet.instruction, prompt)
                    for question in sheet.questions:
                        self.assertIn(question, prompt)

    def test_free_prompt_keeps_profile_context_and_slot_order(self):
        sheet = profiled_sheet()
        prompt = build_prompt(sheet, list(range(sheet.n_items)), language="ch", contract="free")
        self.assertNotIn("测试专属量表", prompt)
        self.assertIn("情境：某人做了一件事。", prompt)
        self.assertIn("请评价第一条。", prompt)
        self.assertNotIn("1. 请评价第一条。", prompt)
        self.assertIn("请评价第三条。", prompt)
        self.assertNotIn("答案类型", prompt)
        self.assertNotIn("输出格式", prompt)

    def test_free_prompt_preserves_source_section_instructions(self):
        sheet = profiled_sheet_with_source_sections()
        prompt = build_prompt(sheet, list(range(sheet.n_items)), language="ch", contract="free")
        self.assertIn("I. 你认为每种观点有多重要？", prompt)
        self.assertIn("II. 你认为每种结果有多重要？", prompt)
        self.assertIn("III. 如果你处在这个情境中，你会怎么做？", prompt)
        self.assertNotIn("作答槽位", prompt)
        self.assertNotIn("JSON", prompt)


class FreeParserTests(unittest.TestCase):
    def slots(self, n=3):
        return profiled_sheet().slots[:n]

    def test_plain_scores_are_read_in_display_order(self):
        raw = "1. 5\n2. 3\n3. 7"
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.status, "ok")
        self.assertEqual([a.score for a in parsed.answers], [5.0, 3.0, 7.0])

    def test_unlabelled_sequence_is_not_recovered_in_ordinary_free_mode(self):
        parsed = parse_free_answers("5 3 7", self.slots())
        self.assertEqual(parsed.status, "unparsed")
        self.assertTrue(all(answer.score is None for answer in parsed.answers))

    def test_scores_only_recovers_complete_unlabelled_sequence(self):
        parsed = parse_free_answers(
            "5 3 7", self.slots(), allow_positional_fallback=True
        )
        self.assertEqual(parsed.status, "ok")
        self.assertEqual([a.score for a in parsed.answers], [5.0, 3.0, 7.0])
        self.assertEqual(parsed.recovery, FREE_PARSER_VERSION + "-positional-v1")

    def test_positional_recovery_requires_exact_item_count(self):
        parsed = parse_free_answers(
            "5 3", self.slots(), allow_positional_fallback=True
        )
        self.assertEqual(parsed.status, "unparsed")
        self.assertTrue(all(answer.score is None for answer in parsed.answers))

    def test_positional_recovery_does_not_collapse_ranges(self):
        parsed = parse_free_answers(
            "5-7 3 7", self.slots(), allow_positional_fallback=True
        )
        self.assertEqual(parsed.status, "unparsed")
        self.assertTrue(all(answer.score is None for answer in parsed.answers))

    def test_scores_with_explanations_are_read(self):
        raw = (
            "1. 5 —— 我不太会为了团队牺牲个人利益。\n"
            "2. 3 —— 有时候会，看情况。\n"
            "3. 7 —— 和谐很重要。"
        )
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.status, "ok")
        self.assertEqual([a.score for a in parsed.answers], [5.0, 3.0, 7.0])

    def test_numbers_in_rationale_are_not_taken_as_scores(self):
        raw = "1. 我有 5 个理由支持这个判断，但无法给出量表分数。\n2. 这取决于具体情境。\n3. 看不出明确分数。"
        parsed = parse_free_answers(raw, self.slots())
        self.assertTrue(all(answer.score is None for answer in parsed.answers))
        self.assertTrue(all(answer.parse_status == "unparsed" for answer in parsed.answers))
        self.assertIn("5 个理由", parsed.answers[0].answer)

    def test_explicit_score_marker_is_read(self):
        raw = "1. 评分：5，因为这个判断较为重要。\n2. score = 3 (moderate)\n3. 我给 7 分。"
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.status, "ok")
        self.assertEqual([answer.score for answer in parsed.answers], [5.0, 3.0, 7.0])

    def test_labelled_and_markdown_wrapped_scores_are_read(self):
        raw = "1. 奶奶应该考虑家人的意愿：3\n2. 毕业后分手可能性：**45%**\n3. 亚洲人：**4 = 中立**"
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.status, "ok")
        self.assertEqual([answer.score for answer in parsed.answers], [3.0, 45.0, 4.0])

    def test_uncertain_ranges_are_preserved_without_midpoint_guessing(self):
        raw = "1. 5-7\n2. 6or7\n3. 2 到 4"
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.status, "partial")
        self.assertEqual([a.score for a in parsed.answers], [None, None, None])
        self.assertEqual(
            [a.answer for a in parsed.answers], ["5-7", "6or7", "2 到 4"]
        )
        self.assertEqual(
            [a.parse_status for a in parsed.answers],
            ["range_response", "range_unresolved", "range_response"],
        )

    def test_open_text_items_keep_prose_and_never_receive_numeric_scores(self):
        slots = [
            ScaleSlot(
                slot_id="story1", question="故事一", response_type="text",
                metadata={"open_ended": True},
            ),
            ScaleSlot(
                slot_id="story2", question="故事二", response_type="text",
                metadata={"open_ended": True},
            ),
        ]
        raw = "1. 应当考虑五个人的生命，因为情境中有 5 人。\n2. 无法判断，理由有 3 点。"
        parsed = parse_free_answers(raw, slots)
        self.assertEqual(parsed.status, "ok")
        self.assertEqual([answer.score for answer in parsed.answers], [None, None])
        self.assertEqual(
            [answer.parse_status for answer in parsed.answers],
            ["text_response", "text_response"],
        )
        self.assertIn("5 人", parsed.answers[0].answer)

    def test_ambiguous_choice_mentions_are_not_treated_as_selection(self):
        slots = [
            ScaleSlot(slot_id="choice1", question="最终选择", response_type="choice", choices=["A", "B"]),
        ]
        parsed = parse_free_answers("1. A 和 B 都有道理，无法确定。", slots)
        self.assertEqual(parsed.status, "unparsed")
        self.assertIsNone(parsed.answers[0].score)
        self.assertEqual(parsed.answers[0].answer, "A 和 B 都有道理，无法确定。")

    def test_more_uncertain_range_forms_are_not_scored(self):
        raw = "1. 5或6\n2. 3/4\n3. between 2 and 4"
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual([a.score for a in parsed.answers], [None, None, None])
        self.assertEqual(
            [a.parse_status for a in parsed.answers],
            ["range_unresolved", "range_unresolved", "range_response"],
        )

    def test_json_range_is_preserved_without_being_scored(self):
        raw = '{"answers":[{"display_index":1,"answer":"3-4"},'
        raw += '{"display_index":2,"answer":5},{"display_index":3,"answer":6}]}'
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(parsed.answers[0].answer, "3-4")
        self.assertIsNone(parsed.answers[0].score)
        self.assertEqual(parsed.answers[0].parse_status, "range_response")
        self.assertEqual(parsed.answers[1].score, 5.0)

    def test_unreadable_items_are_kept_as_unscored_not_dropped(self):
        raw = "1. 5\n2. 说不好，要看具体情况。\n3. 7"
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.status, "partial")
        self.assertEqual([a.score for a in parsed.answers], [5.0, None, 7.0])
        # The raw wording survives so the response can be re-coded later.
        self.assertEqual(parsed.answers[1].answer, "说不好，要看具体情况。")
        self.assertEqual(parsed.answers[1].parse_status, "unparsed")

    def test_response_with_no_readable_score_reports_unparsed(self):
        raw = "1. 这取决于具体情境。\n2. 也说不好。\n3. 看情况。"
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.status, "unparsed")
        self.assertTrue(all(a.score is None for a in parsed.answers))

    def test_response_classification_separates_refusal_and_empty_response(self):
        self.assertEqual(classify_model_response("抱歉，我无法进行评价。"), "refusal")
        self.assertEqual(classify_model_response("This is not a score."), "unparsed")
        self.assertEqual(classify_model_response(""), "empty_response")

    def test_out_of_range_values_are_flagged_but_retained(self):
        raw = "1. 9\n2. 3\n3. 7"
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.answers[0].score, 9.0)
        self.assertEqual(parsed.answers[0].parse_status, "out_of_range")
        self.assertIn("above the declared maximum", parsed.answers[0].parse_error)

    def test_continuation_lines_stay_with_their_item(self):
        raw = "1. 5\n这一点很重要\n2. 3\n3. 7"
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.answers[0].score, 5.0)
        self.assertIn("这一点很重要", parsed.answers[0].answer)

    def test_choices_are_located_in_prose(self):
        slots = [
            ScaleSlot(slot_id="c1", question="最终选择？", response_type="choice",
                      choices=["A", "B"]),
            ScaleSlot(slot_id="c2", question="评分？", response_type="integer",
                      minimum=1, maximum=7),
        ]
        parsed = parse_free_answers("1. 我选 A，因为能救五个人。\n2. 6", slots)
        self.assertEqual(parsed.answers[0].answer, "A")
        self.assertEqual(parsed.answers[1].score, 6.0)

    def test_explicit_choice_with_two_options_is_not_forced(self):
        slots = [
            ScaleSlot(slot_id="c1", question="最终选择？", response_type="choice", choices=["A", "B"]),
        ]
        parsed = parse_free_answers("1. 我选 A 和 B，无法确定。", slots)
        self.assertEqual(parsed.status, "unparsed")
        self.assertIsNone(parsed.answers[0].score)
        self.assertEqual(parsed.answers[0].answer, "我选 A 和 B，无法确定。")

    def test_wrapped_numeric_choice_and_image_label_are_read(self):
        slots = [
            ScaleSlot(slot_id="c1", question="图示选择", response_type="choice", choices=[str(i) for i in range(1, 8)]),
            ScaleSlot(slot_id="c2", question="图示选择", response_type="choice", choices=[str(i) for i in range(1, 8)]),
        ]
        parsed = parse_free_answers("1. 图 **(2)**\n2. 同事：图(7)", slots)
        self.assertEqual(parsed.status, "ok")
        self.assertEqual([answer.answer for answer in parsed.answers], ["2", "7"])


class FreeParserSelfFormattedTests(unittest.TestCase):
    """Models sometimes structure their answer even when nothing asks them to.

    Those responses are fully readable, so discarding them would inflate the
    missing rate for a reason that has nothing to do with the model's answer.
    """

    def slots(self, n=3):
        return profiled_sheet().slots[:n]

    def test_bare_json_response_is_read(self):
        raw = ('{"answers":[{"display_index":1,"answer":5},'
               '{"display_index":2,"answer":3},{"display_index":3,"answer":7}]}')
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.status, "ok")
        self.assertEqual([a.score for a in parsed.answers], [5.0, 3.0, 7.0])
        self.assertEqual(parsed.recovery, "free_text_json_fallback")

    def test_fenced_json_response_is_read(self):
        raw = ('```json\n{"answers":[{"display_index":1,"answer":5},'
               '{"display_index":2,"answer":3},{"display_index":3,"answer":7}]}\n```')
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.status, "ok")
        self.assertEqual([a.score for a in parsed.answers], [5.0, 3.0, 7.0])

    def test_markdown_table_response_is_read(self):
        raw = "| 题号 | 分数 |\n|---|---|\n| 1 | 5 |\n| 2 | 3 |\n| 3 | 7 |"
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.status, "ok")
        self.assertEqual([a.score for a in parsed.answers], [5.0, 3.0, 7.0])

    def test_table_with_extra_columns_keeps_the_score_column(self):
        raw = ("| 题号 | 分数 | 理由 |\n|---|---|---|\n"
               "| 1 | 5 | 不同意 |\n| 2 | 3 | 中立 |\n| 3 | 7 | 同意 |")
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual([a.score for a in parsed.answers], [5.0, 3.0, 7.0])

    def test_table_with_question_before_score_uses_labeled_score_column(self):
        raw = (
            "| 题号 | 题目 | 分数 | 说明 |\n|---|---|---|---|\n"
            "| 1 | 第一条陈述 | 5 = 同意 | 核心内容 |\n"
            "| 2 | 第二条陈述 | 3 = 中立 | 补充说明 |\n"
            "| 3 | 第三条陈述 | 7 = 非常同意 | 其他 |\n"
            "\n总分：15，平均分：5.0"
        )
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.status, "ok")
        self.assertEqual([a.score for a in parsed.answers], [5.0, 3.0, 7.0])

    def test_table_with_question_text_in_first_cell_is_read(self):
        raw = (
            "| 题号/题目 | 分数 | 说明 |\n|---|---|---|\n"
            "| 1. 第一条陈述 | 5 = 同意 | 核心内容 |\n"
            "| 2. 第二条陈述 | 3 = 中立 | 补充说明 |\n"
            "| 3. 第三条陈述 | 7 = 非常同意 | 其他 |"
        )
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.status, "ok")
        self.assertEqual([a.score for a in parsed.answers], [5.0, 3.0, 7.0])

    def test_decimal_in_table_first_cell_is_not_an_item_label(self):
        raw = "| 1.5 | 5 |\n| 2. 第二条陈述 | 3 |"
        parsed = parse_free_answers(raw, self.slots()[:2])
        self.assertEqual(parsed.answers[0].parse_status, "missing")
        self.assertEqual(parsed.answers[1].score, 3.0)

    def test_chinese_yes_no_aliases_are_normalized(self):
        slots = [
            ScaleSlot(slot_id="c1", question="第一问", response_type="choice",
                      choices=["YES", "NO"]),
            ScaleSlot(slot_id="c2", question="第二问", response_type="choice",
                      choices=["YES", "NO"]),
        ]
        parsed = parse_free_answers("1. 是\n2. 否", slots)
        self.assertEqual(parsed.status, "ok")
        self.assertEqual([a.answer for a in parsed.answers], ["YES", "NO"])

    def test_numeric_choice_is_also_available_as_score(self):
        slots = [
            ScaleSlot(slot_id="c1", question="第一问", response_type="choice",
                      choices=["3", "2", "1", "-1", "-2", "-3"]),
        ]
        parsed = parse_free_answers("1. -2", slots)
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(parsed.answers[0].answer, "-2")
        self.assertEqual(parsed.answers[0].score, -2.0)

    def test_prose_without_any_number_is_still_unparsed(self):
        raw = "1. 这取决于情境。\n2. 也说不好。\n3. 看情况。"
        parsed = parse_free_answers(raw, self.slots())
        self.assertEqual(parsed.status, "unparsed")
        self.assertEqual(parsed.recovery, FREE_PARSER_VERSION)

    def test_result_counts_separate_raw_answer_readability_and_scores(self):
        sheet = profiled_sheet()
        raw = "1. 5\n2. 3-4\n3. 这取决于情境。"
        parsed = parse_free_answers(raw, sheet.slots)
        payload = build_result_payload(
            scale_name="fixture",
            language="ch",
            provider="mock",
            model="mock-model",
            temperature=0.0,
            order_id=1,
            seed=None,
            order=[0, 1, 2],
            shuffle_enabled=False,
            sheet=sheet,
            answers_original=parsed.answers,
            parse_result=parsed,
            raw_response=raw,
            prompt="请作答。",
            prompt_contract="free",
        )
        self.assertEqual(payload["answered_item_count"], 3)
        self.assertEqual(payload["read_item_count"], 2)
        self.assertEqual(payload["scored_item_count"], 1)
        self.assertFalse(payload["all_items_read"])
        self.assertTrue(payload["items"][1]["answer_present"])
        self.assertFalse(payload["items"][1]["score_present"])
        self.assertEqual(payload["items"][1]["range_lower"], 3)
        self.assertEqual(payload["items"][1]["range_upper"], 4)

    def test_all_choice_response_is_complete_without_numeric_scores(self):
        slots = [
            ScaleSlot(slot_id="c1", question="第一问", response_type="choice",
                      choices=["A", "B"]),
            ScaleSlot(slot_id="c2", question="第二问", response_type="choice",
                      choices=["A", "B"]),
        ]
        parsed = parse_free_answers("1. 我选 A。\n2. 我选 B。", slots)
        self.assertEqual(parsed.status, "ok")
        self.assertEqual([a.answer for a in parsed.answers], ["A", "B"])


class ConditionIsolationTests(unittest.TestCase):
    def test_legacy_contract_aliases_share_the_source_prompt_condition(self):
        scale = ScaleFile(path=Path("no_such_file.xlsx"), name="测试量表", en=None, ch=None)
        sheet = flat_sheet()
        generation = GenerationConfig.from_mapping({"temperature": 0.0},
                                                   fallback_temperature=1.0)
        strict = BatchRunner([], RunnerConfig(prompt_contract="strict"))
        free = BatchRunner([], RunnerConfig(prompt_contract="free"))
        common = (scale, sheet, "ch", "qwen", "qwen-test", generation)
        self.assertEqual(strict._condition_hash(*common), free._condition_hash(*common))
        scores = BatchRunner([], RunnerConfig(prompt_contract="free_scores_only"))
        self.assertNotEqual(scores._condition_hash(*common), free._condition_hash(*common))
        self.assertEqual(strict._condition_hash(*common), strict._condition_hash(*common))

    def test_descriptor_records_the_contract(self):
        generation = GenerationConfig.from_mapping({"temperature": 0.0},
                                                   fallback_temperature=1.0)
        strict = build_condition_descriptor("controlled", generation,
                                            prompt_contract="strict")
        free = build_condition_descriptor("controlled", generation,
                                          prompt_contract="free")
        self.assertEqual(strict["prompt_contract"], "unconstrained")
        self.assertEqual(free["prompt_contract"], "unconstrained")
        self.assertEqual(strict, free)


class FreePresetTests(unittest.TestCase):
    def test_main_preset_uses_temperature_top_p_and_thinking_off(self):
        preset = apply_preset("free_response_v1")
        self.assertEqual(preset["name"], "主分析")
        self.assertEqual(preset["prompt_config"], {"contract": "unconstrained", "cultural_identity": "none"})
        generation = preset["generation_config"]
        self.assertEqual(generation["temperature"], 0.0)
        self.assertEqual(generation["top_p"], 1.0)
        self.assertEqual(generation["thinking_mode"], "disabled")
        for field in ( "max_output_tokens", "top_k", "min_p", "seed"):
            self.assertIsNone(generation[field], field)

    def test_five_approved_presets_are_registered(self):
        presets = {preset["id"]: preset for preset in list_presets()}
        self.assertEqual(
            set(presets),
            {
                "free_response_v1",
                "free_response_temperature_1_v1",
                "free_response_scores_only_v1",
                "china_identity_v1",
                "usa_identity_v1",
            },
        )
        self.assertEqual(
            presets["free_response_temperature_1_v1"]["prompt_config"],
            {"contract": "unconstrained", "cultural_identity": "none"},
        )
        self.assertEqual(
            presets["free_response_scores_only_v1"]["prompt_config"],
            {"contract": "free_scores_only", "cultural_identity": "none"},
        )

    def test_preset_registry_has_an_extensible_validated_shape(self):
        required = {
            "id",
            "name",
            "description",
            "condition_mode",
            "prompt_config",
            "generation_config",
            "execution_config",
            "warnings",
        }
        for preset in list_presets():
            with self.subTest(preset=preset["id"]):
                self.assertTrue(required.issubset(preset))
                validate_preset_definition(preset["id"], preset)

    def test_parameter_guide_documents_the_contract(self):
        ids = {item["id"] for item in parameter_guide()}
        self.assertIn("prompt_condition", ids)


if __name__ == "__main__":
    unittest.main()
