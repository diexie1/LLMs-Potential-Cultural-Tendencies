"""Offline regression checks for free answer attribution and preservation."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.prompting import parse_free_answers, parse_kohlberg_free_answers
from app.scale_loader import ScaleSlot
from scripts.reparse_results import reparse_result


def rating(question, ident="q1", **kwargs):
    return ScaleSlot(slot_id=ident, question=question, response_type="integer",
                     minimum=1, maximum=7, **kwargs)


def ios_slots():
    return [ScaleSlot(slot_id=f"ios_{relation}", question=f"{index}. Relationship with {relation}",
                      response_type="ios_pair", choices=[str(i) for i in range(1, 8)])
            for index, relation in enumerate(("colleague", "family", "relative", "friend"), 1)]


class ResponseDecoderReviewTests(unittest.TestCase):
    def test_inline_emphasized_score_after_source_question(self):
        for raw in ("1. Alpha statement. **4 = neutral**", "1. Alpha statement.**4**"):
            self.assertEqual(parse_free_answers(raw, [rating("1. Alpha statement.")]).answers[0].score, 4)

    def test_inline_numeric_choice_after_source_question(self):
        slot = ScaleSlot(slot_id="q1", question="1. Alpha statement.", response_type="choice", choices=["-3", "-2", "-1", "1", "2", "3"])
        self.assertEqual(parse_free_answers("1. Alpha statement. **-2**", [slot]).answers[0].score, -2)

    def test_choice_reason_and_does_not_erase_a_leading_score(self):
        slot = ScaleSlot(slot_id="q1", question="1. Alpha statement.", response_type="choice", choices=["-3", "-2", "-1", "1", "2", "3"])
        self.assertEqual(parse_free_answers("1. -1（同意和不同意的理由都存在）", [slot]).answers[0].score, -1)

    def test_numeric_choice_still_rejects_leading_alternatives(self):
        slot = ScaleSlot(slot_id="q1", question="Statement", response_type="choice", choices=["-1", "1"])
        self.assertIsNone(parse_free_answers("1. -1 or 1", [slot]).answers[0].score)

    def test_unlisted_zero_is_retained_and_explained(self):
        slot = ScaleSlot(slot_id="q1", question="Statement", response_type="choice", choices=["-1", "1"])
        answer = parse_free_answers("1. Score: 0", [slot]).answers[0]
        self.assertEqual(answer.parse_status, "invalid_choice")
        self.assertIn("0", answer.answer)
        self.assertIsNone(answer.score)

    def test_nested_question_number_is_not_a_choice_value(self):
        slot = ScaleSlot(slot_id="q1", question="9. Alpha, because of beta.", response_type="choice", choices=["-2", "2"])
        answer = parse_free_answers("1. **9. Alpha, because of beta.** → **-2**", [slot]).answers[0]
        self.assertEqual(answer.score, -2)

    def test_multiline_source_numbers_are_preserved_after_shuffle(self):
        slots = [rating("2. Beta statement\nResponse options", "q2"), rating("1. Alpha statement\nResponse options", "q1")]
        result = parse_free_answers("2. Beta statement: 6\n1. 3", slots)
        self.assertEqual([a.score for a in result.answers], [6, 3])
        self.assertEqual(result.answers[1].mapping_method, "source_number")

    def test_bulleted_score_belongs_to_preceding_question(self):
        result = parse_free_answers("1. Alpha statement\n- Score: **4 = neutral**", [rating("1. Alpha statement")])
        self.assertEqual(result.answers[0].score, 4)

    def test_circled_selection_after_arrow_or_score_label(self):
        for raw in ("1. Alpha statement — **③ Occasionally**", "1. Alpha statement\n- Score: **③ Occasionally**"):
            self.assertEqual(parse_free_answers(raw, [rating("1. Alpha statement")]).answers[0].score, 3)

    def test_option_codebook_is_not_a_response(self):
        raw = "1. Alpha statement\n① Never = 1\n② Rarely = 2\n③ Sometimes = 3\n④ Often = 4"
        answer = parse_free_answers(raw, [rating("1. Alpha statement")]).answers[0]
        self.assertEqual(answer.parse_status, "option_codebook")
        self.assertIsNone(answer.score)

    def test_explicit_final_estimate_resolves_discussed_range(self):
        slot = ScaleSlot(slot_id="count", question="Count", response_type="integer", minimum=0, maximum=100)
        for raw, score in (("1. About 10–15 out of 100\nSo I'd estimate roughly **12**.", 12),
                           ("1. 约10-20人\n这里我给出一个整数：**15**。", 15),
                           ("1. 约10-20人\n若必须填整数，我填 **15**。", 15)):
            self.assertEqual(parse_free_answers(raw, [slot]).answers[0].score, score)

    def test_natural_rating_and_estimate_with_sentence_period(self):
        for raw, score in (("1. I'd rate it **1 = unacceptable**.", 1),
                           ("1. Research reports 10–30%; I'd estimate **20**.", 20),
                           ("1. 我估计 **85** 人。", 85)):
            slot = ScaleSlot(slot_id="count", question="Count", response_type="integer", minimum=0, maximum=100)
            self.assertEqual(parse_free_answers(raw, [slot]).answers[0].score, score)

    def test_different_explicit_estimates_still_conflict(self):
        answer = parse_free_answers("1. I'd estimate 2.\nMy estimate is 3.", [rating("Question")]).answers[0]
        self.assertEqual(answer.parse_status, "conflicting_values")

    def test_short_unique_source_subject_ignores_changed_number(self):
        slots = [rating("1. 您对甲乙的总体评价是？", "q1"), rating("2. 您对丙丁的总体评价是？", "q2")]
        result = parse_free_answers("1. 丙丁：4\n2. 甲乙：2", slots)
        self.assertEqual([a.score for a in result.answers], [2, 4])

    def test_short_chinese_question_can_identify_unnumbered_answer(self):
        slots = [rating("1. 甲乙人", "q1"), rating("2. 丙丁人", "q2")]
        result = parse_free_answers("- 丙丁人：4\n- 甲乙人：3", slots)
        self.assertEqual([a.score for a in result.answers], [3, 4])

    def test_repeated_roman_sections_keep_inline_scores(self):
        slots = [rating("1. Alpha statement", "human_rights_e01_viewpoint_01", context_id="human_rights_e01", section_id="viewpoint"),
                 rating("2. Beta statement", "human_rights_e01_viewpoint_02", context_id="human_rights_e01", section_id="viewpoint")]
        result = parse_free_answers("**<Episode 1>**\nI. Alpha statement **4**\nI. Beta statement **5**", slots)
        self.assertEqual([a.score for a in result.answers], [4, 5])

    def test_summary_table_uses_its_episode_and_subquestion_columns(self):
        slots = [rating("1. Alpha", "human_rights_e01_action_01", context_id="human_rights_e01", section_id="action"),
                 rating("1. Beta", "human_rights_e02_action_01", context_id="human_rights_e02", section_id="action")]
        raw = "Episode 2\nIII. Actions\n1. Beta: 4\n总体汇总\n| 情境 | 题目 | 评分 |\n|---|---|---|\n| 1 | III-1 Alpha | 3 |\n| 2 | III-1 Beta | 4 |"
        self.assertEqual([a.score for a in parse_free_answers(raw, slots).answers], [3, 4])

    def test_logical_natural_decisions_are_not_truth_score_aliases(self):
        slot = ScaleSlot(slot_id="reasoning_concrete_01", question="Conclusion", response_type="choice", choices=["YES", "NO"], section_id="logical_validity")
        for value, expected in (("Valid: the stated premises imply the conclusion.", "YES"),
                                ("Conclusion follows logically.", "YES"),
                                ("The conclusion does not follow.", "NO"),
                                ("逻辑：是", "YES"), ("逻辑判断：不能推出。", "NO"),
                                ("所以不能推出。", "NO"), ("+3（逻辑上成立）", None),
                                ("一般情况分数：这个结论在现实中成立。+3", None)):
            with self.subTest(value=value):
                answer = parse_free_answers("1. " + value, [slot]).answers[0]
                if expected:
                    self.assertEqual(answer.answer, expected)
                else:
                    self.assertEqual(answer.parse_status, "unparsed")

    def test_linked_tasks_can_match_conclusion_on_later_line(self):
        slots = [ScaleSlot(slot_id="reasoning_concrete_01", question="1.\nConclusion: All alpha are beta", response_type="choice", choices=["YES", "NO"], section_id="logical_validity"),
                 ScaleSlot(slot_id="reasoning_belief_01", question="1. All alpha are beta", response_type="integer", minimum=-3, maximum=3, section_id="conclusion_believability", linked_slot_id="reasoning_concrete_01")]
        raw = "1.\nConclusion: All alpha are beta\n- Premise 1: All alpha are beta\n- 逻辑：是\n- 一般情况分数：+2"
        result = parse_free_answers(raw, slots)
        self.assertEqual(result.answers[0].answer, "YES")
        self.assertEqual(result.answers[1].score, 2)

    def test_general_intro_does_not_switch_combined_task_to_belief_only(self):
        slots = [ScaleSlot(slot_id="reasoning_concrete_01", question="1.\nConclusion: All alpha are beta", response_type="choice", choices=["YES", "NO"], section_id="logical_validity"),
                 ScaleSlot(slot_id="reasoning_belief_01", question="1. All alpha are beta", response_type="integer", minimum=-3, maximum=3, section_id="conclusion_believability", linked_slot_id="reasoning_concrete_01")]
        raw = "我会按前面的逻辑要求和后面的一般情况分数要求分别作答：\n1. All alpha are beta\n判断：是\n分数：+2"
        result = parse_free_answers(raw, slots)
        self.assertEqual(result.answers[0].answer, "YES")
        self.assertEqual(result.answers[1].score, 2)

    def test_combined_task_table_reads_each_typed_answer_column(self):
        slots = [ScaleSlot(slot_id="reasoning_concrete_01", question="1.\nConclusion: All alpha are beta", response_type="choice", choices=["YES", "NO"], section_id="logical_validity"),
                 ScaleSlot(slot_id="reasoning_belief_01", question="1. All alpha are beta", response_type="integer", minimum=-3, maximum=3, section_id="conclusion_believability", linked_slot_id="reasoning_concrete_01")]
        raw = "| 题号 | 结论 | 逻辑能否推出 | 一般情况分数 |\n|---|---|---|---|\n| 1 | All alpha are beta | 是 | -2 |"
        result = parse_free_answers(raw, slots)
        self.assertEqual(result.answers[0].answer, "YES")
        self.assertEqual(result.answers[1].score, -2)

    def test_score_then_explicit_logical_decision_reads_both(self):
        slots = [ScaleSlot(slot_id="reasoning_concrete_01", question="1.\nConclusion: All alpha are beta", response_type="choice", choices=["YES", "NO"], section_id="logical_validity"),
                 ScaleSlot(slot_id="reasoning_belief_01", question="1. All alpha are beta", response_type="integer", minimum=-3, maximum=3, section_id="conclusion_believability", linked_slot_id="reasoning_concrete_01")]
        result = parse_free_answers("1. All alpha are beta\n- +3 (definitely true) — follows logically from premises", slots)
        self.assertEqual(result.answers[0].answer, "YES")
        self.assertEqual(result.answers[1].score, 3)

    def test_empty_numeric_response_with_explicit_refusal_is_classified(self):
        from app.prompting import classify_model_response
        raw = "作为人工智能，我应当保持客观中立，不对任何领导人进行主观评分或价值判断。"
        self.assertEqual(classify_model_response(raw, parse_free_answers(raw, [rating("Question")])), "refusal")

    def test_plain_score_label_still_reads_without_emphasis(self):
        self.assertEqual(parse_free_answers("1. Score: 4", [rating("Question")]).answers[0].score, 4)

    def test_final_list_is_separate_from_last_item(self):
        result = parse_free_answers("1. Alpha statement: 4\nFinal list:\n1. Alpha statement: 4", [rating("1. Alpha statement")])
        self.assertEqual(result.answers[0].score, 4)

    def test_unique_conclusion_subject_establishes_local_numbering(self):
        slots = [ScaleSlot(slot_id="reasoning_concrete_02", question="2.\nConclusion: All rabbits sleep", response_type="choice", choices=["YES", "NO"], section_id="logical_validity"),
                 ScaleSlot(slot_id="reasoning_concrete_01", question="1.\nConclusion: All birds sing", response_type="choice", choices=["YES", "NO"], section_id="logical_validity")]
        raw = "Logical validity first:\n1. Premise: birds migrate. Invalid.\n2. Valid."
        result = parse_free_answers(raw, slots)
        self.assertEqual([a.answer for a in result.answers], ["YES", "NO"])

    def test_truth_task_numbering_does_not_fill_ambiguous_logic_rows(self):
        slots = [ScaleSlot(slot_id="reasoning_concrete_02", question="2.\nConclusion: All rabbits sleep", response_type="choice", choices=["YES", "NO"], section_id="logical_validity"),
                 ScaleSlot(slot_id="reasoning_concrete_01", question="1.\nConclusion: All birds sing", response_type="choice", choices=["YES", "NO"], section_id="logical_validity"),
                 ScaleSlot(slot_id="reasoning_belief_02", question="2. All rabbits sleep", response_type="integer", minimum=-3, maximum=3, section_id="conclusion_believability", linked_slot_id="reasoning_concrete_02"),
                 ScaleSlot(slot_id="reasoning_belief_01", question="1. All birds sing", response_type="integer", minimum=-3, maximum=3, section_id="conclusion_believability", linked_slot_id="reasoning_concrete_01")]
        raw = "Logical validity first:\n1. YES\n2. NO\nTruth scores:\n1. All birds sing: 2\n2. All rabbits sleep: -2"
        result = parse_free_answers(raw, slots)
        self.assertTrue(all(a.parse_status == "missing" for a in result.answers[:2]))

    def test_ios_pair_label_and_rationale_are_decoded(self):
        raw = ("1. Colleagues: Pair (2). Two circles interact and have separate lives.\n"
               "2. Family members: (6) Close, dependent, and encompassing.\n"
               "3. Relatives: Pair (4). Shared bond and common background.\n"
               "4. Friends: Select (3). They share interests and maintain distinct lives.")
        self.assertEqual([a.score for a in parse_free_answers(raw, ios_slots()).answers], [2, 6, 4, 3])

    def test_ios_chinese_or_in_rationale_does_not_reject_selection(self):
        raw = "3. 亲戚：有血缘或姻亲联系，对应图 (4)（重叠适中）。"
        result = parse_free_answers(raw, ios_slots())
        self.assertEqual(result.answers[2].score, 4)

    def test_ios_pair_late_in_sentence_and_named_summary_agree(self):
        raw = "1. Colleagues: Work relationships are independent. Pair (3) is most appropriate.\nSummary:\n1. Colleagues — (3)"
        self.assertEqual(parse_free_answers(raw, ios_slots()).answers[0].score, 3)

    def test_ios_relation_labels_ignore_other_relations_in_rationale(self):
        raw = "2. Relatives: Pair (4). They share family history but less contact than friends.\n3. Friends: Pair (5). Closer than colleagues and relatives."
        result = parse_free_answers(raw, [ios_slots()[3], ios_slots()[2], ios_slots()[1], ios_slots()[0]])
        self.assertEqual([a.score for a in result.answers], [5, 4, None, None])

    def test_ios_relatively_is_not_a_relative_relationship_label(self):
        raw = "1. Colleagues are relatively independent — (3)."
        result = parse_free_answers(raw, ios_slots())
        self.assertEqual(result.answers[0].score, 3)
        self.assertIsNone(result.answers[2].score)

    def test_ios_unnumbered_relationship_lines_are_mapped(self):
        for raw in ("同事：图(2)\n家人：图(6)\n亲戚：图(4)\n朋友：图(3)",
                    "Colleagues: Pair (2)\nFamily members: Pair (6)\nRelatives: Pair (4)\nFriends: Pair (3)"):
            with self.subTest(raw=raw):
                self.assertEqual([a.score for a in parse_free_answers(raw, ios_slots()).answers], [2, 6, 4, 3])

    def test_ios_multiple_diagram_candidates_remain_unresolved(self):
        for value in ("Pair (5) or (6)", "(6) (or 7)", "图(5)或图(6)", "图4/图5", "(3)—(4)", "图(3)，图(4)"):
            with self.subTest(value=value):
                self.assertIsNone(parse_free_answers(f"2. Family members: {value}", ios_slots()).answers[1].score)

    def test_ios_note_with_unused_diagrams_does_not_change_selection(self):
        raw = "4. Friends: Pair (4).\n(Note: figure (2) is distant and figure (7) is closest.)"
        self.assertEqual(parse_free_answers(raw, ios_slots()).answers[3].score, 4)

    def test_ios_compact_multirelationship_summary_is_not_last_items_score(self):
        raw = ("1. 同事：图(1)\n2. 家庭成员：图(7)\n3. 亲戚：图(3)\n4. 朋友：图(5)\n"
               "同事 - (1)，家庭成员 - (7)，亲戚 - (3)，朋友 - (5)。")
        self.assertEqual([a.score for a in parse_free_answers(raw, ios_slots()).answers], [1, 7, 3, 5])

    def test_ios_invalid_diagram_numbers_are_not_partial_matches(self):
        for value in ("Pair (14)", "Pair (4.5)", "Pair (-4)", "Pair (8)"):
            with self.subTest(value=value):
                self.assertIsNone(parse_free_answers(f"1. Colleagues: {value}", ios_slots()).answers[0].score)

    def test_original_numbers_after_shuffle_follow_question_text(self):
        slots = [rating("2. Beta question", "q2"), rating("1. Alpha question", "q1")]
        result = parse_free_answers("2. Beta question: 6\n1. Alpha question: 3", slots)
        self.assertEqual([a.score for a in result.answers], [6, 3])

    def test_generated_display_numbers_follow_question_text(self):
        slots = [rating("2. Beta question", "q2"), rating("1. Alpha question", "q1")]
        result = parse_free_answers("1. Beta question: 6\n2. Alpha question: 3", slots)
        self.assertEqual([a.score for a in result.answers], [6, 3])

    def test_bare_numbers_after_shuffle_are_not_guessed(self):
        slots = [rating("2. Beta question", "q2"), rating("1. Alpha question", "q1")]
        result = parse_free_answers("1. 6\n2. 3", slots)
        self.assertTrue(all(a.score is None for a in result.answers))

    def test_aligned_source_and_display_numbers_are_unambiguous(self):
        slots = [rating("1. Alpha question"), rating("2. Beta question", "q2")]
        result = parse_free_answers("1. 6\n2. 3", slots)
        self.assertEqual([a.score for a in result.answers], [6, 3])

    def test_identical_questions_in_unscoped_contexts_do_not_pick_first(self):
        slots = [rating("1. Shared question?", "q1", context_id="s1"),
                 rating("1. Shared question?", "q2", context_id="s2")]
        result = parse_free_answers("1. Shared question?\n5", slots)
        self.assertTrue(all(a.score is None for a in result.answers))

    def test_long_question_wins_over_nested_short_question(self):
        slots = [rating("How often do you pray?"),
                 rating("How often do you pray spontaneously when inspired?", "q2")]
        result = parse_free_answers("1. How often do you pray spontaneously when inspired?\n③ Occasionally", slots)
        self.assertIsNone(result.answers[0].score)
        self.assertEqual(result.answers[1].score, 3)

    def test_contradictory_number_modes_do_not_fill_ambiguous_item(self):
        slots = [rating("2. Beta question", "q2"), rating("1. Alpha question", "q1"),
                 rating("4. Delta question", "q4"), rating("3. Gamma question", "q3")]
        result = parse_free_answers("1. Beta question: 5\n2. Beta question: 5\n3. 6", slots)
        self.assertIsNone(result.answers[2].score)
        self.assertIsNone(result.answers[3].score)

    def test_unknown_context_does_not_assign_into_known_context(self):
        slot = rating("1. A statement", "human_rights_e01_viewpoint_01", context_id="human_rights_e01", section_id="viewpoint")
        result = parse_free_answers("Episode 99\nI. Importance\n1. 5", [slot])
        self.assertIsNone(result.answers[0].score)

    def test_item_body_keeps_original_punctuation(self):
        result = parse_free_answers("1. 说不好，要看情境。\n第二行：保留‘文字’。", [rating("Question")])
        self.assertEqual(result.answers[0].answer, "说不好，要看情境。\n第二行：保留‘文字’。")

    def test_explicit_score_not_invalidated_by_rationale_range(self):
        result = parse_free_answers("1. 6\n解释：常见的评分范围是 4–7。", [rating("Question")])
        self.assertEqual(result.answers[0].score, 6)

    def test_percentage_range_in_rationale_does_not_override_scalar(self):
        slot = ScaleSlot(slot_id="prob", question="Probability", response_type="number", minimum=0, maximum=100)
        result = parse_free_answers("1. 80\n调查中约70%–90%，这里估计80。", [slot])
        self.assertEqual(result.answers[0].score, 80)

    def test_answer_percentage_range_is_not_collapsed(self):
        slot = ScaleSlot(slot_id="prob", question="Probability", response_type="number", minimum=0, maximum=100)
        result = parse_free_answers("1. 40%–60%", [slot])
        self.assertIsNone(result.answers[0].score)
        self.assertEqual(result.answers[0].parse_status, "range_response")
        self.assertEqual((result.answers[0].range_lower, result.answers[0].range_upper,
                          result.answers[0].range_unit), (40, 60, "%"))

    def test_closed_interval_formats_and_units(self):
        slot = ScaleSlot(slot_id="q1", question="Question", response_type="number", minimum=-10, maximum=100)
        for text, lower, upper, unit in (("约10%—20%", 10, 20, "%"),
                                        ("-3--1", -3, -1, ""),
                                        ("between -3 and -1", -3, -1, ""),
                                        ("from .2 to .6", .2, .6, ""),
                                        ("介于2和4之间", 2, 4, ""),
                                        ("[3,5]", 3, 5, ""),
                                        ("评分：3至5", 3, 5, ""),
                                        ("40 to 60 percent", 40, 60, "%")):
            with self.subTest(text=text):
                result = parse_free_answers("1. " + text, [slot])
                answer = result.answers[0]
                self.assertEqual(result.status, "ok")
                self.assertEqual(answer.parse_status, "range_response")
                self.assertEqual((answer.range_lower, answer.range_upper, answer.range_unit), (lower, upper, unit))
                self.assertIsNone(answer.score)
                self.assertEqual(answer.answer, text)

    def test_repeated_same_interval_keeps_both_passages(self):
        slot = rating("1. Question")
        answer = parse_free_answers("1. Question: 3–5\n1. Question: about 3–5", [slot]).answers[0]
        self.assertEqual(answer.parse_status, "range_response")
        self.assertEqual((answer.range_lower, answer.range_upper), (3, 5))
        self.assertIn("about 3–5", answer.answer)

    def test_different_intervals_or_scalar_and_interval_conflict(self):
        slot = rating("1. Question")
        for raw in ("1. Question: 3–5\n1. Question: 4–6",
                    "1. Question: 3–5\n1. Question: 4"):
            with self.subTest(raw=raw):
                answer = parse_free_answers(raw, [slot]).answers[0]
                self.assertEqual(answer.parse_status, "conflicting_answers")
                self.assertIsNone(answer.score)
                self.assertIn("3–5", answer.answer)

    def test_interval_bounds_are_retained_without_clipping_or_swapping(self):
        for text, status, lower, upper in (("0–9", "range_out_of_range", 0, 9),
                                           ("5–3", "range_unresolved", 5, 3)):
            with self.subTest(text=text):
                answer = parse_free_answers("1. " + text, [rating("Question")]).answers[0]
                self.assertEqual(answer.parse_status, status)
                self.assertEqual((answer.range_lower, answer.range_upper), (lower, upper))
                self.assertIsNone(answer.score)

    def test_numeric_choice_interval_does_not_invent_missing_choices(self):
        slot = ScaleSlot(slot_id="q1", question="Question", response_type="choice",
                         choices=["-3", "-2", "-1", "1", "2", "3"])
        for text, status in (("-3到-1", "range_response"), ("-1到1", "invalid_choice")):
            with self.subTest(text=text):
                answer = parse_free_answers("1. " + text, [slot]).answers[0]
                self.assertEqual(answer.parse_status, status)
                self.assertIsNotNone(answer.range_lower)
                self.assertIsNone(answer.score)

    def test_discrete_candidates_and_slash_are_not_intervals(self):
        for text in ("3or4", "3或4", "3/4", "3–5 or 6–7", "40% or 60%",
                     "3–5–7", "3–5分或6–7分", "40%或60%", "40%/60%"):
            with self.subTest(text=text):
                answer = parse_free_answers("1. " + text, [rating("Question")]).answers[0]
                self.assertIsNone(answer.range_lower)
                self.assertIsNone(answer.score)

    def test_json_percentage_alternatives_are_not_one_number_or_interval(self):
        answer = parse_free_answers('{"answers":[{"display_index":1,"answer":"40%或60%"}]}', [rating("Question")]).answers[0]
        self.assertIsNone(answer.score)
        self.assertIsNone(answer.range_lower)

    def test_open_text_range_remains_only_text(self):
        slot = ScaleSlot(slot_id="q1", question="Question", response_type="text", metadata={"open_ended": True})
        answer = parse_free_answers("3–5", [slot]).answers[0]
        self.assertEqual(answer.parse_status, "text_response")
        self.assertIsNone(answer.range_lower)

    def test_final_summary_can_explicitly_resolve_interval(self):
        slot = rating("1. Question")
        answer = parse_free_answers("1. Question: 3–5\nFinal list:\n1. Question: 4", [slot]).answers[0]
        self.assertEqual(answer.score, 4)
        self.assertIsNone(answer.range_lower)
        self.assertIn("3–5", answer.answer)

    def test_json_and_table_intervals_are_read_without_scalar_scores(self):
        slot = ScaleSlot(slot_id="q1", question="Question", response_type="number", minimum=0, maximum=100)
        for raw in ('{"answers":[{"display_index":1,"answer":"40%–60%"}]}',
                    '| No | Score |\n|---|---|\n| 1 | 40%–60% |'):
            with self.subTest(raw=raw):
                answer = parse_free_answers(raw, [slot]).answers[0]
                self.assertEqual(answer.parse_status, "range_response")
                self.assertEqual((answer.range_lower, answer.range_upper, answer.range_unit), (40, 60, "%"))
                self.assertIsNone(answer.score)

    def test_reparsed_interval_exports_fields_and_completion_counts(self):
        import csv
        import tempfile
        from app.runner import BatchRunner
        item = {"index": 1, "question": "Question", **rating("Question").metadata_dict()}
        old = {"items": [item], "shuffle_order": [1], "raw_response": "1. 3–5"}
        result = reparse_result(old)
        self.assertEqual(result["read_item_count"], 1)
        self.assertEqual(result["scored_item_count"], 0)
        self.assertTrue(result["all_items_read"])
        self.assertEqual(result["items"][0]["range_text"], "3–5")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.csv"
            BatchRunner._write_csv(path, result)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual((row["range_text"], row["range_lower"], row["range_upper"]), ("3–5", "3.0", "5.0"))
        self.assertEqual(row["score"], "")
        self.assertEqual(row["trial_raw_response"], old["raw_response"])

    def test_dash_inside_prose_range_is_not_an_answer_separator(self):
        for raw in ("1. 通常评3—5分。", "1. 常见评分为 3 — 5。", "1. 概率大概 40%—60%。"):
            with self.subTest(raw=raw):
                answer = parse_free_answers(raw, [rating("Question")]).answers[0]
                self.assertIsNone(answer.score)

    def test_explicit_people_count_is_read(self):
        slot = ScaleSlot(slot_id="count", question="Estimated count", response_type="integer", minimum=0, maximum=100)
        result = parse_free_answers("1. 估计人数：约20人\n解释中可能介于10–30人。", [slot])
        self.assertEqual(result.answers[0].score, 20)

    def test_decimal_is_not_rounded_for_integer_item(self):
        result = parse_free_answers("1. 3.5", [rating("Question")])
        self.assertIsNone(result.answers[0].score)
        self.assertEqual(result.answers[0].parse_status, "non_integer")

    def test_fraction_with_scale_denominator_means_score(self):
        result = parse_free_answers("1. 2/7", [rating("Question")])
        self.assertEqual(result.answers[0].score, 2)

    def test_other_fraction_does_not_choose_first_value(self):
        result = parse_free_answers("1. 2/3", [rating("Question")])
        self.assertIsNone(result.answers[0].score)

    def test_table_keeps_nonnumeric_x_selection(self):
        slot = ScaleSlot(slot_id="choice", question="Opinion", response_type="choice", choices=["-2", "-1", "0", "1", "2", "X"])
        result = parse_free_answers("| Item | Answer |\n|---|---|\n| 1 | X |", [slot])
        self.assertEqual(result.answers[0].answer, "X")
        self.assertIsNone(result.answers[0].score)
        self.assertEqual(result.status, "ok")

    def test_all_circled_options_are_not_a_selected_answer(self):
        result = parse_free_answers("1. Question\n① Never\n② Rarely\n③ Sometimes\n④ Often", [rating("Question")])
        self.assertIsNone(result.answers[0].score)
        self.assertEqual(result.answers[0].parse_status, "conflicting_values")

    def test_table_circled_codebook_on_one_line_is_not_a_selection(self):
        answer = parse_free_answers("| Item | Score |\n|---|---|\n| 1 | ①0 ②1 ③2 ④3 ⑤4 |", [rating("Question")]).answers[0]
        self.assertIsNone(answer.score)

    def test_nested_source_number_is_not_read_as_score(self):
        answer = parse_free_answers("1. **12. A source statement.** → **5** (Agree)", [rating("12. A source statement.")]).answers[0]
        self.assertEqual(answer.score, 5)

    def test_aggregate_list_after_answer_does_not_override_last_score(self):
        answer = parse_free_answers("1. 3\n即：\n5，4，3，4，3", [rating("Question")]).answers[0]
        self.assertEqual(answer.score, 3)

    def test_conflicting_repeated_scores_stay_unscored(self):
        result = parse_free_answers("1. 5\n1. 6", [rating("Question")])
        self.assertIsNone(result.answers[0].score)
        self.assertEqual(result.answers[0].parse_status, "conflicting_answers")

    def test_repeated_range_and_scalar_are_both_retained(self):
        for raw in ("1. 5–6\n1. 7", "1. 7\n1. 5–6"):
            with self.subTest(raw=raw):
                answer = parse_free_answers(raw, [rating("Question")]).answers[0]
                self.assertIsNone(answer.score)
                self.assertIn("5–6", answer.answer)
                self.assertIn("7", answer.answer)

    def test_choice_explanation_and_does_not_mean_second_selection(self):
        slot = ScaleSlot(slot_id="c", question="Question", response_type="choice", choices=["-1", "1", "2"])
        answer = parse_free_answers("1. 1 — Agree slightly. People hesitate and dislike stealing.", [slot]).answers[0]
        self.assertEqual(answer.answer, "1")

    def test_choice_after_arrow_is_read(self):
        slot = ScaleSlot(slot_id="c", question="1. A source question", response_type="choice", choices=["-2", "1"])
        answer = parse_free_answers("1. A source question → -2 (Disagree)", [slot]).answers[0]
        self.assertEqual(answer.score, -2)

    def test_unparsed_echo_does_not_hide_explicit_answer(self):
        answer = parse_free_answers("1. Question\n1. 5", [rating("Question")]).answers[0]
        self.assertEqual(answer.score, 5)

    def test_final_named_summary_can_resolve_range_given_by_model(self):
        slot = ScaleSlot(slot_id="p", question="Chance of breakup?", response_type="number", maximum=100, context_id="items")
        raw = "1. Chance of breakup?\n40%–50%\nSummary of answers:\n1. Chance of breakup?: 45%"
        answer = parse_free_answers(raw, [slot]).answers[0]
        self.assertEqual(answer.score, 45)
        self.assertIn("40%–50%", answer.answer)

    def test_model_can_explicitly_choose_one_value_after_range_in_same_item(self):
        slot = ScaleSlot(slot_id="p", question="Probability", response_type="number", maximum=100)
        answer = parse_free_answers("1. 40%–50%\n一般可给45%。", [slot]).answers[0]
        self.assertEqual(answer.score, 45)

    def test_final_numbered_summary_resolves_fixed_item_range(self):
        answer = parse_free_answers("1. 5–6\n最终答案：\n1. 5", [rating("Question")]).answers[0]
        self.assertEqual(answer.score, 5)

    def test_partial_question_matches_support_consistent_complete_section_table(self):
        slots = [rating("A source statement", "attribution_s01_a01", context_id="attribution_s01", section_id="attribution_agreement"),
                 rating("A longer statement about the environment", "attribution_s01_a02", context_id="attribution_s01", section_id="attribution_agreement")]
        result = parse_free_answers("Scenario 1\nA. Scores\n| Statement | Score |\n|---|---|\n| A source statement | 5 |\n| Environment | 3 |", slots)
        self.assertEqual([a.score for a in result.answers], [5, 3])

    def test_summary_numbering_cannot_borrow_votes_from_previous_segment(self):
        slots = [rating("2. Beta question", "q2"), rating("1. Alpha question", "q1")]
        raw = "2. Beta question: 6\n1. Alpha question: 3\nSummary:\n1. 5\n2. 7"
        result = parse_free_answers(raw, slots)
        self.assertEqual([a.score for a in result.answers], [6, 3])

    def test_question_mentioning_number_is_not_itself_an_answer(self):
        slot = rating("I give 3 people help.")
        answer = parse_free_answers("1. I give 3 people help.", [slot]).answers[0]
        self.assertIsNone(answer.score)

    def test_logical_and_belief_tasks_remain_separate(self):
        slots = [ScaleSlot(slot_id="reasoning_concrete_01", question="1.\nPremise 1: All fish need water\nConclusion: All fish like water", response_type="choice", choices=["YES", "NO"], section_id="logical_validity"),
                 ScaleSlot(slot_id="reasoning_belief_01", question="1. All fish like water", response_type="integer", minimum=-3, maximum=3, section_id="conclusion_believability")]
        raw = "Logical validity first:\n1. All fish like water — Valid\nNow scoring based on general real-world truth:\n1. All fish like water → +2"
        result = parse_free_answers(raw, slots)
        self.assertEqual(result.answers[0].answer, "YES")
        self.assertEqual(result.answers[1].score, 2)

    def test_linked_logical_and_belief_answers_can_share_one_response_record(self):
        slots = [ScaleSlot(slot_id="reasoning_concrete_01", question="1.\nConclusion: All fish like water", response_type="choice", choices=["YES", "NO"], section_id="logical_validity"),
                 ScaleSlot(slot_id="reasoning_belief_01", question="1. All fish like water", response_type="integer", minimum=-3, maximum=3, section_id="conclusion_believability", linked_slot_id="reasoning_concrete_01")]
        result = parse_free_answers("1. All fish like water\n判断：是\n分数：+2", slots)
        self.assertEqual(result.answers[0].answer, "YES")
        self.assertEqual(result.answers[1].score, 2)

    def test_linked_score_cannot_supply_a_missing_logical_judgment(self):
        slots = [ScaleSlot(slot_id="reasoning_concrete_01", question="1.\nConclusion: All fish like water", response_type="choice", choices=["YES", "NO"], section_id="logical_validity"),
                 ScaleSlot(slot_id="reasoning_belief_01", question="1. All fish like water", response_type="integer", minimum=-3, maximum=3, section_id="conclusion_believability", linked_slot_id="reasoning_concrete_01")]
        result = parse_free_answers("1. All fish like water\n分数：+2", slots)
        self.assertIsNone(result.answers[0].answer)
        self.assertEqual(result.answers[1].score, 2)

    def test_logical_no_in_statement_is_not_a_selected_no(self):
        slot = ScaleSlot(slot_id="reasoning_concrete_01", question="1.\nConclusion: No cats are blue", response_type="choice", choices=["YES", "NO"], section_id="logical_validity")
        answer = parse_free_answers("1. No cats are blue — Valid", [slot]).answers[0]
        self.assertEqual(answer.answer, "YES")

    def test_logical_yes_or_no_remains_unresolved(self):
        slot = ScaleSlot(slot_id="reasoning_concrete_01", question="Conclusion", response_type="choice", choices=["YES", "NO"], section_id="logical_validity")
        answer = parse_free_answers("1. YES or NO", [slot]).answers[0]
        self.assertEqual(answer.parse_status, "unparsed")

    def test_task_words_in_rationale_do_not_switch_task(self):
        slots = [ScaleSlot(slot_id="reasoning_belief_01", question="1. First statement", response_type="integer", section_id="conclusion_believability"),
                 ScaleSlot(slot_id="reasoning_belief_02", question="2. Second statement", response_type="integer", section_id="conclusion_believability")]
        result = parse_free_answers("Task 2: Truth scores\n1. 2\nThis is independent of logical validity and belongs to truth scoring.\n2. 3", slots)
        self.assertEqual([a.score for a in result.answers], [2, 3])

    def test_single_open_item_keeps_entire_numbered_response(self):
        slot = ScaleSlot(slot_id="open", question="Discuss", metadata={"open_ended": True})
        raw = "1. 第一部分：有3个原因。\n2. 第二部分，不强制数字。"
        answer = parse_free_answers(raw, [slot]).answers[0]
        self.assertEqual(answer.answer, raw)
        self.assertIsNone(answer.score)

    def test_merged_interview_stories_follow_source_rows(self):
        slots = [ScaleSlot(slot_id="a", question="Story\n1. Should the doctor help the patient?", metadata={"open_ended": True}),
                 ScaleSlot(slot_id="b", question="Continuation\n3. Should the judge punish the doctor?", metadata={"open_ended": True}),
                 ScaleSlot(slot_id="c", question="New story\n1. Should Judy tell her mother?", metadata={"open_ended": True})]
        raw = "Situation 1\n1. Should the doctor help the patient?\nYes, because…\n3. Should the judge punish the doctor?\nConsider the circumstances.\nSituation 2\n1. Should Judy tell her mother?\nShe should be honest."
        result = parse_kohlberg_free_answers(raw, slots)
        self.assertEqual(result.status, "ok")
        self.assertIn("judge", result.answers[1].answer)
        self.assertNotIn("Judy", result.answers[1].answer)
        self.assertIn("Judy", result.answers[2].answer)
        self.assertTrue(all(a.score is None for a in result.answers))

    def test_interview_unique_first_question_prefix_recovers_minor_edit(self):
        slots = [ScaleSlot(slot_id="a", question="Story\n3. 法官应该判处海因兹某种刑罚，还是应该终止审判？", metadata={"open_ended": True})]
        raw = "情境三：法官\n3. 法官应该判处海因兹某种刑罚，还是终止审判？\n应从轻处理。"
        result = parse_kohlberg_free_answers(raw, slots)
        self.assertIn("应从轻处理。", result.answers[0].answer)
        self.assertEqual(result.answers[0].mapping_method, "interview_question_prefix")

    def test_offline_reparse_keeps_snapshots_and_canonical_order(self):
        slots = [rating("1. Alpha question", "q1"), rating("2. Beta question", "q2")]
        items = [{"index": i + 1, "question": s.question, **s.metadata_dict(), "score": None} for i, s in enumerate(slots)]
        old = {"items": items, "shuffle_order": [2, 1], "raw_response": "2. Beta question: 6\n1. Alpha question: 3",
               "prompt": "Saved prompt", "request_snapshot": {"model": "fixture"}, "response_snapshot": {"id": "fixture"},
               "blocks": [{"slots": copy.deepcopy(items)}]}
        snapshot = copy.deepcopy(old)
        result = reparse_result(old)
        self.assertEqual(old, snapshot)
        self.assertEqual(result["scores"], [3, 6])
        self.assertEqual([i["score"] for i in result["blocks"][0]["slots"]], [3, 6])
        for key in ("raw_response", "prompt", "shuffle_order", "request_snapshot", "response_snapshot"):
            self.assertEqual(result[key], old[key])

    def test_invalid_saved_permutation_is_rejected(self):
        item = {"question": "Question", **rating("Question").metadata_dict()}
        with self.assertRaises(ValueError):
            reparse_result({"items": [item], "shuffle_order": [2]})


if __name__ == "__main__":
    unittest.main()
