"""Schema-aware layouts for research instruments.

The workbook convention used by most scales is deliberately simple: A1 is a
title, A2 is an instruction, and each subsequent cell is one response item.
Some instruments in this module contain scenarios, section headers, images,
and/or several response slots.  Others use the ordinary flat workbook layout
but still need an explicit answer type and valid range.  Treating either group
as untyped text would make output parsing scientifically unsafe.

This module is intentionally declarative at the boundary: it converts an Excel
sheet into typed ``ScaleTaskBlock`` / ``ScaleSlot`` objects.  Prompting,
randomisation, parsing, and result writing use those objects rather than
trying to infer structure from model prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from .scale_loader import ScaleImage, ScaleSheet, ScaleSlot, ScaleTaskBlock


class ProfileBuildError(ValueError):
    """The workbook does not meet the documented structure for its profile."""


@dataclass
class ProfileLayout:
    """The validated structured layout for one language sheet."""

    profile_id: str
    profile_version: str
    profile_label: str
    instruction: str
    blocks: List[ScaleTaskBlock]
    default_shuffle: bool = False
    shuffle_supported: bool = True


@dataclass(frozen=True)
class ResponseRule:
    """Declarative validation rule for one flat response slot."""

    response_type: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Tuple[str, ...] = ()
    slot_id: str = ""
    metadata: Tuple[Tuple[str, object], ...] = ()


@dataclass(frozen=True)
class TypedFlatSpec:
    """Validated schema for an otherwise ordinary A1/A2/A3+ workbook."""

    profile_id: str
    label: str
    rules: Tuple[ResponseRule, ...]
    default_shuffle: bool = True
    shuffle_supported: bool = True
    shuffle_policy: str = "within"


def _repeat_rule(
    count: int,
    response_type: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    choices: Tuple[str, ...] = (),
) -> Tuple[ResponseRule, ...]:
    return tuple(
        ResponseRule(
            response_type=response_type,
            minimum=minimum,
            maximum=maximum,
            choices=choices,
        )
        for _ in range(count)
    )


def _dilemma_rules(include_other_probability: bool = False) -> Tuple[ResponseRule, ...]:
    rules = [
        ResponseRule("choice", choices=("A", "B"), slot_id="final_choice"),
        ResponseRule("integer", 1, 7, slot_id="moral_acceptability"),
        ResponseRule("integer", 1, 7, slot_id="choice_confidence"),
        ResponseRule("integer", 0, 100, slot_id="estimated_choice_a_count"),
    ]
    if include_other_probability:
        rules.append(
            ResponseRule(
                "integer",
                0,
                100,
                slot_id="estimated_other_choice_a_probability",
            )
        )
    return tuple(rules)


def _uniform_spec(
    profile_id: str,
    label: str,
    count: int,
    minimum: int,
    maximum: int,
) -> TypedFlatSpec:
    return TypedFlatSpec(
        profile_id=profile_id,
        label=label,
        rules=_repeat_rule(count, "integer", minimum, maximum),
    )


TYPED_FLAT_SPECS: Dict[str, TypedFlatSpec] = {
    "individual_collective_priority_v1": _uniform_spec(
        "individual_collective_priority_v1", "个体－集体优先性量表（1–7）", 5, 1, 7
    ),
    "humanism_normativism_v1": _uniform_spec(
        "humanism_normativism_v1", "人文主义与规范主义量表（1–7）", 30, 1, 7
    ),
    "human_rights_attitudes_v1": _uniform_spec(
        "human_rights_attitudes_v1", "人权态度量表（1–5）", 21, 1, 5
    ),
    "human_nature_philosophy_v1": TypedFlatSpec(
        profile_id="human_nature_philosophy_v1",
        label="人类本性哲学量表（3/2/1/−1/−2/−3）",
        rules=tuple(
            ResponseRule(
                "choice",
                choices=("3", "2", "1", "-1", "-2", "-3"),
            )
            for _ in range(20)
        ),
    ),
    "religiosity_centrality_v1": TypedFlatSpec(
        profile_id="religiosity_centrality_v1",
        label="宗教中心性量表（CRSi-20）",
        rules=tuple(
            ResponseRule("integer", 1, 8 if 8 <= item_index <= 10 else 5)
            for item_index in range(20)
        ),
    ),
    "war_attitudes_v1": _uniform_spec(
        "war_attitudes_v1", "战争态度量表（1–6）", 26, 1, 6
    ),
    "popular_democracy_definition_v1": _uniform_spec(
        "popular_democracy_definition_v1", "民主的通俗定义量表（1–10）", 10, 1, 10
    ),
    "democracy_cognition_v1": _uniform_spec(
        "democracy_cognition_v1", "民主感知量表（1–5）", 35, 1, 5
    ),
    "short_sdo_v1": _uniform_spec(
        "short_sdo_v1", "简版社会支配倾向量表（1–10）", 4, 1, 10
    ),
    "crowd_impression_v1": _uniform_spec(
        "crowd_impression_v1", "人群印象量表（1–7）", 4, 1, 7
    ),
    "religion_impression_v1": _uniform_spec(
        "religion_impression_v1", "宗教印象量表（1–7）", 5, 1, 7
    ),
    "human_good_evil_v1": _uniform_spec(
        "human_good_evil_v1", "性善性恶量表（1–7）", 2, 1, 7
    ),
    "war_justification_v1": _uniform_spec(
        "war_justification_v1", "战争正当性评估（1–7）", 2, 1, 7
    ),
    "leader_evaluation_v1": _uniform_spec(
        "leader_evaluation_v1", "领导者评价（1–7）", 3, 1, 7
    ),
    "personal_cultural_values_v1": _uniform_spec(
        "personal_cultural_values_v1", "个人文化价值观量表（1–7）", 6, 1, 7
    ),
    "collectivism_v1": _uniform_spec(
        "collectivism_v1", "集体主义量表（1–7）", 10, 1, 7
    ),
    "change_expectation_v1": TypedFlatSpec(
        profile_id="change_expectation_v1",
        label="变化预期任务（0–100）",
        rules=_repeat_rule(4, "number", 0, 100),
    ),
    "modern_racism_v1": TypedFlatSpec(
        profile_id="modern_racism_v1",
        label="现代种族主义量表（−2/−1/0/1/2/X）",
        rules=tuple(
            ResponseRule(
                "choice",
                choices=("-2", "-1", "0", "1", "2", "X"),
                metadata=(("valid_nonresponse_choices", ["X"]),),
            )
            for _ in range(8)
        ),
    ),
    "trolley_dilemma_v1": TypedFlatSpec(
        profile_id="trolley_dilemma_v1",
        label="电车困境（选择＋判断）",
        rules=_dilemma_rules(),
        default_shuffle=False,
        shuffle_supported=False,
        shuffle_policy="fixed",
    ),
    "footbridge_dilemma_v1": TypedFlatSpec(
        profile_id="footbridge_dilemma_v1",
        label="天桥困境（选择＋判断）",
        rules=_dilemma_rules(),
        default_shuffle=False,
        shuffle_supported=False,
        shuffle_policy="fixed",
    ),
    "prisoners_dilemma_v1": TypedFlatSpec(
        profile_id="prisoners_dilemma_v1",
        label="囚徒困境（选择＋判断）",
        rules=_dilemma_rules(include_other_probability=True),
        default_shuffle=False,
        shuffle_supported=False,
        shuffle_policy="fixed",
    ),
}


SCALE_NAME_PROFILES: Dict[str, str] = {
    "个体-集体优先性量表-评分表": "individual_collective_priority_v1",
    "人文主义与规范主义量表-评分表": "humanism_normativism_v1",
    "人权态度量表-评分表": "human_rights_attitudes_v1",
    "修订版人类本性哲学量表-评分表": "human_nature_philosophy_v1",
    "变化预期任务-评分表": "change_expectation_v1",
    "囚徒困境-评分表": "prisoners_dilemma_v1",
    "天桥困境-评分表": "footbridge_dilemma_v1",
    "宗教中心性量表-评分表": "religiosity_centrality_v1",
    "战争态度量表-评分表": "war_attitudes_v1",
    "民主的通俗定义-评分表": "popular_democracy_definition_v1",
    # The current workbook uses the revised display name "民主感知", while
    # the original instrument and its typed schema are named "民主认知".
    "民主感知量表-评分表": "democracy_cognition_v1",
    "民主认知量表-评分表": "democracy_cognition_v1",
    "现代种族主义量表-评分表": "modern_racism_v1",
    "电车困境-评分表": "trolley_dilemma_v1",
    "简版社会支配倾向量表-评分表": "short_sdo_v1",
    "自创-人群印象": "crowd_impression_v1",
    "自创-宗教印象": "religion_impression_v1",
    "自创-性善性恶": "human_good_evil_v1",
    "自创-战争正当性评估": "war_justification_v1",
    "自创-领导者评价": "leader_evaluation_v1",
    "集体主义量表-个人文化价值观-评分表": "personal_cultural_values_v1",
    "集体主义量表-评分表": "collectivism_v1",
    "科尔伯格道德判断访谈-纯原版-评分表": "kohlberg_mji_v1",
    "科尔伯格道德判断访谈-普适版-评分表": "kohlberg_mji_v1",
}


PROFILE_LABELS: Dict[str, str] = {
    "attribution_bias_v1": "归因偏差任务（情境组块）",
    "human_rights_v1": "人权敏感量表（情境 × 观点/结果/行动）",
    "ios_v1": "自我－他人融合量表（圆圈对图示）",
    "intuitive_reasoning_v1": "直觉（vs. 形式）推理任务（逻辑 + 信念评价）",
    "kohlberg_mji_v1": "科尔伯格道德判断访谈（开放式访谈）",
    "kohlberg_story_v1": "科尔伯格单故事开放回答",
}


KOHLBERG_FORM_OPTIONS = {
    "all": "全部三套故事（原表完整内容）",
    "A": "Form A：故事 III、III'、I",
    "B": "Form B：故事 IV、IV'、II",
    "C": "Form C：故事 V、VIII、VII",
}


def normalize_kohlberg_form(value: object) -> str:
    """Normalize the optional MJI form selector used by the workbench."""

    raw = str(value or "all").strip()
    if raw.casefold() == "all":
        return "all"
    upper = raw.upper()
    if upper in {"A", "B", "C"}:
        return upper
    return "all"


def identify_profile(scale_name: str, title: str = "") -> Optional[str]:
    """Return a profile identifier only for known non-flat instruments.

    We use the file name first and title second.  This avoids accidentally
    applying a special parser to an unrelated ordinary workbook merely because
    it contains words such as "situation" or "reasoning".
    """

    named_profile = SCALE_NAME_PROFILES.get(str(scale_name or "").strip())
    if named_profile:
        return named_profile

    text = f"{scale_name} {title}".casefold()
    if "归因偏差" in text or "attribution bias" in text:
        return "attribution_bias_v1"
    if "人权敏感" in text or "human rights sensitivity" in text:
        return "human_rights_v1"
    if ("自我" in text and "融合" in text) or "inclusion of other" in text:
        return "ios_v1"
    if "直觉推理" in text or "intuitive (vs. formal) reasoning" in text:
        return "intuitive_reasoning_v1"
    # Story-level Kohlberg workbooks deliberately carry the instrument name in
    # their visible filename/title, but contain only one dilemma.  Route them
    # to a one-slot open-text profile so the nine-story MJI parser is reserved
    # for the original interview workbook.
    is_kohlberg_story = (
        ("科尔伯格" in text or "kohlberg" in text)
        and ("困境" in text or "dilemma" in text)
        and "道德判断访谈" not in text
        and "moral judgment interview" not in text
    )
    if is_kohlberg_story:
        return "kohlberg_story_v1"
    if "科尔伯格" in text or "kohlberg" in text or "moral judgment interview" in text:
        return "kohlberg_mji_v1"
    return None


def _clean(value: object) -> str:
    return str(value or "").replace("\r", "").replace("\xa0", " ").strip()


def _nonempty(values: Sequence[str]) -> List[str]:
    return [value for value in (_clean(v) for v in values) if value]


_NUMBERED_ITEM_RE = re.compile(
    r"(?ms)^\s*(\d+)\s*[.．、]\s*(.*?)(?=^\s*\d+\s*[.．、]\s*|\Z)"
)


def _numbered_items(text: str) -> List[Tuple[int, str]]:
    """Read explicitly numbered items, retaining wrapped lines as one item."""

    return [
        (int(match.group(1)), _clean(match.group(2)))
        for match in _NUMBERED_ITEM_RE.finditer(_clean(text))
        if _clean(match.group(2))
    ]


def _intro_and_numbered_items(text: str) -> Tuple[str, List[Tuple[int, str]]]:
    """Separate a local instruction from the numbered items that follow it."""

    cleaned = _clean(text)
    first = _NUMBERED_ITEM_RE.search(cleaned)
    if first is None:
        return cleaned, []
    return _clean(cleaned[: first.start()]), _numbered_items(cleaned[first.start() :])


def _expect_numbered(
    text: str,
    expected: int,
    description: str,
) -> List[str]:
    numbered = _numbered_items(text)
    values = [value for _number, value in numbered]
    observed_numbers = [number for number, _value in numbered]
    if len(values) != expected or observed_numbers != list(range(1, expected + 1)):
        raise ProfileBuildError(
            f"{description}: expected {expected} numbered items 1–{expected}, "
            f"found {len(values)} ({observed_numbers})."
        )
    return values


def _profile_instruction(cells: Sequence[str]) -> str:
    """A2 is normally the global instruction; keep it outside answer slots."""

    return _clean(cells[1]) if len(cells) > 1 else ""


def _attribution_groupness_block(text: str) -> ScaleTaskBlock:
    """Build the fixed-order six-item groupness block from its own cell."""
    group_instruction, group_items_raw = _intro_and_numbered_items(_clean(text))
    group_items = _expect_numbered(
        "\n".join(f"{index}. {value}" for index, value in group_items_raw),
        6,
        "Attribution Bias Task, groupness section",
    )
    return ScaleTaskBlock(
        block_id="attribution_groupness",
        label="Groupness ratings",
        context="",
        instruction=group_instruction,
        slots=[
            ScaleSlot(
                slot_id=f"attribution_groupness_g{item_no:02d}",
                question=question,
                response_type="integer",
                minimum=1,
                maximum=7,
                context_id="attribution_groupness",
                section_id="groupness",
                section_label=group_instruction,
                metadata={"subtask": "groupness"},
            )
            for item_no, question in enumerate(group_items, start=1)
        ],
        shuffle_policy="fixed",
    )


def _attribution_layout(cells: Sequence[str], language: str) -> ProfileLayout:
    raw = _nonempty(cells[2:])
    # Two equivalent layouts are accepted:
    #   * 6 cells — the six-item groupness block is embedded after scenario 6;
    #   * 7 cells — 6 scenario cells followed by a separate groupness cell.
    # The 7-cell form keeps the groupness task as its own block without nesting
    # it physically inside scenario 6.
    if len(raw) not in (6, 7):
        raise ProfileBuildError(
            f"Attribution Bias Task: expected 6 or 7 non-empty cells "
            f"(6 scenarios, optionally +1 groupness cell), found {len(raw)}."
        )
    scenario_cells = raw[:6]
    separate_groupness = raw[6] if len(raw) == 7 else None

    marker = {
        "a": re.compile(r"(?m)^\s*A[.．]\s*"),
        "b": re.compile(r"(?m)^\s*B[.．]\s*"),
        "c": re.compile(r"(?m)^\s*C[.．]\s*"),
    }
    blocks: List[ScaleTaskBlock] = []
    group_block: Optional[ScaleTaskBlock] = None

    for scenario_no, raw_scenario in enumerate(scenario_cells, start=1):
        text = _clean(raw_scenario)
        a_match = marker["a"].search(text)
        b_match = marker["b"].search(text)
        c_match = marker["c"].search(text)
        if not (a_match and b_match and c_match and a_match.start() < b_match.start() < c_match.start()):
            if language.casefold().startswith("ch"):
                raise ProfileBuildError(
                    f"归因偏差任务：情境 {scenario_no} 缺少 A/B/C 中的一段。"
                    "该语言版本与完整结构不等价，已阻止运行，避免把情境或说明误当作题项。"
                )
            raise ProfileBuildError(
                "Attribution Bias Task, scenario "
                f"{scenario_no}: A / B / C response sections are all required. "
                "The workbook is not language-equivalent enough to run safely."
            )

        context = _clean(text[: a_match.start()])
        a_body = _clean(text[a_match.end() : b_match.start()])
        b_prompt = _clean(text[b_match.end() : c_match.start()])
        c_rest = _clean(text[c_match.end() :])

        if separate_groupness is None:
            # Legacy layout: the groupness block follows section C after a blank
            # paragraph.  Detected from a blank-line boundary rather than item
            # numbers, because C and groupness both restart numbering at one.
            group_start = re.search(
                r"\n\s*\n(?=(?:Please\s+read\s+each\s+sentence|"
                r"基于一般情况，请为每个选项明确给出一个分数))",
                c_rest,
                re.IGNORECASE,
            )
            if group_start:
                if scenario_no != 6:
                    raise ProfileBuildError(
                        "Attribution Bias Task: groupness items may only follow scenario 6."
                    )
                group_block = _attribution_groupness_block(c_rest[group_start.end() :])
                c_prompt = _clean(c_rest[: group_start.start()])
            else:
                c_prompt = c_rest
        else:
            c_prompt = c_rest

        a_instruction, a_items_raw = _intro_and_numbered_items(a_body)
        a_items = _expect_numbered("\n".join(
            f"{index}. {value}" for index, value in a_items_raw
        ), 10, f"Attribution Bias Task, scenario {scenario_no}, section A")
        if not b_prompt or not c_prompt:
            raise ProfileBuildError(
                f"Attribution Bias Task, scenario {scenario_no}: B/C prompt is empty."
            )

        slots = [
            ScaleSlot(
                slot_id=f"attribution_s{scenario_no:02d}_a{item_no:02d}",
                question=question,
                response_type="integer",
                minimum=1,
                maximum=7,
                context_id=f"attribution_s{scenario_no:02d}",
                section_id="attribution_agreement",
                section_label=a_instruction,
                metadata={"scenario": scenario_no, "subtask": "attribution_rating"},
            )
            for item_no, question in enumerate(a_items, start=1)
        ]
        slots.extend(
            [
                ScaleSlot(
                    slot_id=f"attribution_s{scenario_no:02d}_b",
                    question=b_prompt,
                    response_type="number",
                    minimum=0,
                    maximum=100,
                    context_id=f"attribution_s{scenario_no:02d}",
                    section_id="other_actor_probability",
                    section_label="B",
                    metadata={"scenario": scenario_no, "subtask": "other_actor_probability"},
                ),
                ScaleSlot(
                    slot_id=f"attribution_s{scenario_no:02d}_c",
                    question=c_prompt,
                    response_type="number",
                    minimum=0,
                    maximum=100,
                    context_id=f"attribution_s{scenario_no:02d}",
                    section_id="same_actor_probability",
                    section_label="C",
                    metadata={"scenario": scenario_no, "subtask": "same_actor_probability"},
                ),
            ]
        )
        label = context.split("\n", 1)[0] or f"Scenario {scenario_no}"
        blocks.append(
            ScaleTaskBlock(
                block_id=f"attribution_s{scenario_no:02d}",
                label=label,
                context=context,
                instruction="",
                slots=slots,
                # Context and its 12 answers form an inseparable unit.
                shuffle_policy="block",
                metadata={"scenario": scenario_no},
            )
        )

    if separate_groupness is not None:
        group_block = _attribution_groupness_block(separate_groupness)

    if group_block is None:
        raise ProfileBuildError(
            "Attribution Bias Task: missing the six-item groupness section after scenario 6."
        )
    blocks.append(group_block)
    n_slots = sum(len(block.slots) for block in blocks)
    if n_slots != 78:
        raise ProfileBuildError(f"Attribution Bias Task: expected 78 slots, found {n_slots}.")
    return ProfileLayout(
        profile_id="attribution_bias_v1",
        profile_version="1",
        profile_label=PROFILE_LABELS["attribution_bias_v1"],
        instruction=_profile_instruction(cells),
        blocks=blocks,
        # The six vignettes are independent contextual units. Shuffle only
        # those complete blocks; attribution_groupness remains fixed at the
        # end because it is a separate follow-up task.
        default_shuffle=True,
        shuffle_supported=True,
    )


_ROMAN_SECTION_RE = re.compile(
    r"(?m)^\s*(?P<label>III|II|I|Ⅲ|Ⅱ|Ⅰ)[.．、]\s*"
)


def _human_rights_layout(cells: Sequence[str], language: str) -> ProfileLayout:
    scenario_cells = _nonempty(cells[2:])
    if len(scenario_cells) != 10:
        raise ProfileBuildError(
            f"Human Rights Sensitivity Scale: expected 10 episode cells, found {len(scenario_cells)}."
        )

    section_ids = ("viewpoint", "outcome", "action")
    blocks: List[ScaleTaskBlock] = []
    for episode_no, raw_episode in enumerate(scenario_cells, start=1):
        text = _clean(raw_episode)
        matches = list(_ROMAN_SECTION_RE.finditer(text))
        if len(matches) != 3:
            raise ProfileBuildError(
                f"Human Rights Sensitivity Scale, episode {episode_no}: expected I/II/III sections, found {len(matches)}."
            )
        context = _clean(text[: matches[0].start()])
        slots: List[ScaleSlot] = []
        for section_no, match in enumerate(matches):
            end = matches[section_no + 1].start() if section_no + 1 < len(matches) else len(text)
            content = _clean(text[match.end() : end])
            section_instruction, item_pairs = _intro_and_numbered_items(content)
            items = _expect_numbered(
                "\n".join(f"{index}. {value}" for index, value in item_pairs),
                2,
                f"Human Rights Sensitivity Scale, episode {episode_no}, section {section_no + 1}",
            )
            section_id = section_ids[section_no]
            section_label = f"{match.group('label')}. {section_instruction}".strip()
            slots.extend(
                ScaleSlot(
                    slot_id=f"human_rights_e{episode_no:02d}_{section_id}_{item_no:02d}",
                    question=question,
                    response_type="integer",
                    minimum=1,
                    maximum=5,
                    context_id=f"human_rights_e{episode_no:02d}",
                    section_id=section_id,
                    section_label=section_label,
                    metadata={"episode": episode_no, "section_order": section_no + 1},
                )
                for item_no, question in enumerate(items, start=1)
            )
        label = context.split("\n", 1)[0] or f"Episode {episode_no}"
        blocks.append(
            ScaleTaskBlock(
                block_id=f"human_rights_e{episode_no:02d}",
                label=label,
                context=context,
                slots=slots,
                shuffle_policy="block",
                metadata={"episode": episode_no},
            )
        )

    n_slots = sum(len(block.slots) for block in blocks)
    if n_slots != 60:
        raise ProfileBuildError(f"Human Rights Sensitivity Scale: expected 60 slots, found {n_slots}.")
    return ProfileLayout(
        profile_id="human_rights_v1",
        profile_version="1",
        profile_label=PROFILE_LABELS["human_rights_v1"],
        instruction=_profile_instruction(cells),
        blocks=blocks,
        default_shuffle=True,
        shuffle_supported=True,
    )


def _ios_layout(
    cells: Sequence[str], language: str, images: Sequence[ScaleImage]
) -> ProfileLayout:
    if not images:
        raise ProfileBuildError(
            "Inclusion of Other in the Self Scale: the required seven-diagram image was not extracted."
        )
    raw_questions = _nonempty(cells[3:])
    if len(raw_questions) != 4:
        raise ProfileBuildError(
            f"Inclusion of Other in the Self Scale: expected 4 relationship items, found {len(raw_questions)}."
        )
    relation_ids = ("colleague", "family", "relative", "friend")
    # The embedded IOS plate contains seven pre-drawn diagrams.  A response
    # identifies one complete pair-of-circles diagram by its printed label
    # (1)–(7); it is not a pair of circle numbers and must never be sent
    # through numeric range parsing.
    diagram_choices = [str(index) for index in range(1, 8)]
    slots: List[ScaleSlot] = []
    for item_no, (relation_id, source_question) in enumerate(
        zip(relation_ids, raw_questions), start=1
    ):
        question = source_question
        normalization_note: Optional[str] = None
        if relation_id == "relative" and language.casefold().startswith("en"):
            # Current workbooks correctly use "relative(s)".  Retain support
            # for the older duplicated "friends" source while making that
            # compatibility normalization explicit in the saved result.
            if re.search(r"\brelatives?\b", question, flags=re.IGNORECASE):
                pass
            elif re.search(r"\bfriends?\b", question, flags=re.IGNORECASE):
                question = re.sub(
                    r"\bfriends?\b", "relatives", question, flags=re.IGNORECASE
                )
                normalization_note = (
                    "English source item 3 duplicated 'friends'; rendered as 'relatives' "
                    "to align with the Chinese source and canonical relation slot."
                )
            else:
                raise ProfileBuildError(
                    "Inclusion of Other in the Self Scale: English item 3 does not identify the expected relative relationship."
                )
        slots.append(
            ScaleSlot(
                slot_id=f"ios_{relation_id}",
                question=question,
                response_type="ios_pair",
                choices=diagram_choices,
                context_id="ios_relationships",
                section_id="relationship",
                source_question=source_question,
                normalization_note=normalization_note,
                metadata={
                    "relation": relation_id,
                    "image_required": True,
                    "pair_diagram": True,
                    "score_semantics": (
                        "IOS diagram overlap code 1–7; larger number means greater overlap"
                    ),
                },
            )
        )
    note = _clean(cells[2]) if len(cells) > 2 else ""
    return ProfileLayout(
        profile_id="ios_v1",
        profile_version="3",
        profile_label=PROFILE_LABELS["ios_v1"],
        instruction=note,
        blocks=[
            ScaleTaskBlock(
                block_id="ios_relationships",
                label="Relationship pair-diagram selections",
                context="",
                instruction=note,
                slots=slots,
                # Lu et al. randomized the four relationship questions.  The
                # seven diagrams inside the embedded image are a fixed answer
                # key and are never shuffled.
                shuffle_policy="within",
                metadata={
                    "image_required": True,
                    "pair_diagram_answer_key_fixed": True,
                },
            )
        ],
        default_shuffle=True,
        shuffle_supported=True,
    )


def _intuitive_reasoning_layout(cells: Sequence[str], language: str) -> ProfileLayout:
    values = _nonempty(cells)
    if len(values) < 43:
        raise ProfileBuildError(
            "Intuitive Reasoning Task: workbook is shorter than the documented 24 + 16 structure."
        )
    rating_index: Optional[int] = None
    for index, value in enumerate(values[2:], start=2):
        normalized = value.replace("−", "-")
        if "-3" in normalized and "+3" in normalized and (
            "score" in normalized.casefold() or "分数" in normalized
        ):
            rating_index = index
            break
    if rating_index is None:
        raise ProfileBuildError(
            "Intuitive Reasoning Task: cannot locate the -3…+3 conclusion-rating instruction."
        )
    arguments = values[2:rating_index]
    rating_instruction = values[rating_index]
    ratings = values[rating_index + 1 :]
    if len(arguments) != 24:
        raise ProfileBuildError(
            f"Intuitive Reasoning Task: expected 24 logic arguments, found {len(arguments)}."
        )
    if len(ratings) != 16:
        raise ProfileBuildError(
            f"Intuitive Reasoning Task: expected 16 conclusion ratings, found {len(ratings)}."
        )

    concrete_slots = [
        ScaleSlot(
            slot_id=f"reasoning_concrete_{number:02d}",
            question=question,
            response_type="choice",
            choices=["YES", "NO"],
            context_id="reasoning_concrete",
            section_id="logical_validity",
            metadata={"argument_no": number, "argument_type": "concrete"},
        )
        for number, question in enumerate(arguments[:16], start=1)
    ]
    abstract_slots = [
        ScaleSlot(
            slot_id=f"reasoning_abstract_{number:02d}",
            question=question,
            response_type="choice",
            choices=["YES", "NO"],
            context_id="reasoning_abstract",
            section_id="logical_validity",
            metadata={"argument_no": number + 16, "argument_type": "abstract"},
        )
        for number, question in enumerate(arguments[16:], start=1)
    ]
    rating_slots = [
        ScaleSlot(
            slot_id=f"reasoning_belief_{number:02d}",
            question=question,
            response_type="integer",
            minimum=-3,
            maximum=3,
            context_id="reasoning_belief",
            section_id="conclusion_believability",
            linked_slot_id=f"reasoning_concrete_{number:02d}",
            metadata={"argument_no": number, "linked_task": "concrete_validity"},
        )
        for number, question in enumerate(ratings, start=1)
    ]
    # The source instrument has two sequential tasks: 24 logical-validity
    # judgments followed by 16 conclusion-believability ratings. Shuffle each
    # task internally while keeping the task blocks in this fixed order.
    blocks = [
        ScaleTaskBlock(
            block_id="reasoning_logical_validity",
            label="Logical-validity task",
            context="",
            slots=concrete_slots + abstract_slots,
            shuffle_policy="within",
        ),
        ScaleTaskBlock(
            block_id="reasoning_belief",
            label="Conclusion-believability task",
            context="",
            instruction=rating_instruction,
            slots=rating_slots,
            shuffle_policy="within",
        ),
    ]
    return ProfileLayout(
        profile_id="intuitive_reasoning_v1",
        profile_version="2",
        profile_label=PROFILE_LABELS["intuitive_reasoning_v1"],
        instruction=_profile_instruction(values),
        blocks=blocks,
        default_shuffle=True,
        shuffle_supported=True,
    )


def _kohlberg_story_layout(cells: Sequence[str], language: str) -> ProfileLayout:
    """Represent one split Kohlberg dilemma as one open-ended response slot."""

    raw = _nonempty(cells[2:])
    if len(raw) != 1:
        raise ProfileBuildError(
            "Kohlberg story subscale: expected exactly one story cell, "
            f"found {len(raw)} cells."
        )
    slot = ScaleSlot(
        slot_id="kohlberg_story",
        question=raw[0],
        response_type="text",
        context_id="kohlberg_story",
        section_id="dilemma",
        metadata={"open_ended": True, "story_subscale": True},
    )
    return ProfileLayout(
        profile_id="kohlberg_story_v1",
        profile_version="1",
        profile_label=PROFILE_LABELS["kohlberg_story_v1"],
        instruction=_profile_instruction(cells),
        blocks=[
            ScaleTaskBlock(
                block_id="kohlberg_story",
                label="Kohlberg story",
                slots=[slot],
                shuffle_policy="fixed",
                metadata={"story_subscale": True},
            )
        ],
        default_shuffle=False,
        shuffle_supported=False,
    )


def _kohlberg_mji_layout(cells: Sequence[str], language: str) -> ProfileLayout:
    """Load either one named MJI form or the legacy combined workbook."""

    raw = _nonempty(cells[2:])
    title = _clean(cells[0]) if cells else ""
    form_match = re.search(r"\bform\s*([abc])\b", title, re.IGNORECASE)
    standalone_form = form_match.group(1).upper() if form_match else None

    if standalone_form:
        expected_stories = {"A": 4, "B": 4, "C": 3}[standalone_form]
        if len(raw) != expected_stories:
            raise ProfileBuildError(
                f"Kohlberg MJI Form {standalone_form}: expected "
                f"{expected_stories} situations, found {len(raw)}."
            )
        form_stories = [
            (standalone_form, str(index), story)
            for index, story in enumerate(raw, start=1)
        ]
        version = f"2-standalone-form-{standalone_form}"
        label = f"{PROFILE_LABELS['kohlberg_mji_v1']}（Form {standalone_form}）"
    else:
        if len(raw) != 10:
            raise ProfileBuildError(
                "Kohlberg MJI: expected 9 dilemma cells plus 1 scoring-note cell, "
                f"found {len(raw)} cells."
            )
        form_map = (
            ("A", "III"),
            ("A", "III'"),
            ("A", "I"),
            ("B", "IV"),
            ("B", "IV'"),
            ("B", "II"),
            ("C", "V"),
            ("C", "VIII"),
            ("C", "VII"),
        )
        form_stories = [
            (form, dilemma, story)
            for (form, dilemma), story in zip(form_map, raw[:9])
        ]
        version = "1"
        label = PROFILE_LABELS["kohlberg_mji_v1"]

    blocks: List[ScaleTaskBlock] = []
    for index, (form, dilemma, story) in enumerate(form_stories, start=1):
        slot_id = (
            f"kohlberg_{form.casefold()}_situation_{index:02d}"
            if standalone_form
            else f"kohlberg_dilemma_{index:02d}"
        )
        block_label = (
            f"Form {form} · Situation {index}"
            if standalone_form
            else f"Form {form} · Dilemma {dilemma}"
        )
        slot = ScaleSlot(
            slot_id=slot_id,
            question=story,
            response_type="text",
            context_id=f"kohlberg_{form.casefold()}",
            section_id="dilemma",
            metadata={
                "form": form,
                "dilemma": dilemma,
                "open_ended": True,
                "standalone_form": bool(standalone_form),
            },
        )
        blocks.append(
            ScaleTaskBlock(
                block_id=f"kohlberg_{form.casefold()}_{index:02d}",
                label=block_label,
                slots=[slot],
                shuffle_policy="fixed",
                metadata={
                    "form": form,
                    "dilemma": dilemma,
                    "standalone_form": bool(standalone_form),
                },
            )
        )

    # MJI is an interview: preserve the complete source interview sequence,
    # including Form C's otherwise independent stories.
    return ProfileLayout(
        profile_id="kohlberg_mji_v1",
        profile_version=version,
        profile_label=label,
        instruction=_profile_instruction(cells),
        blocks=blocks,
        default_shuffle=False,
        shuffle_supported=False,
    )


def kohlberg_standalone_form(sheet: ScaleSheet) -> Optional[str]:
    """Return A/B/C when a workbook already contains exactly one MJI form."""

    if sheet.profile_id != "kohlberg_mji_v1" or not sheet.blocks:
        return None
    if not all(block.metadata.get("standalone_form") for block in sheet.blocks):
        return None
    forms = {str(block.metadata.get("form") or "").upper() for block in sheet.blocks}
    return next(iter(forms)) if len(forms) == 1 else None


def select_kohlberg_form(sheet: ScaleSheet, form: object = "all") -> ScaleSheet:
    """Return a view of an MJI sheet containing all stories or one form."""

    if sheet.profile_id != "kohlberg_mji_v1":
        return sheet
    # The updated source uses one file per form; its filename/source content
    # already determines the form, so legacy selectors must not filter it.
    if kohlberg_standalone_form(sheet):
        return sheet
    selected = normalize_kohlberg_form(form)
    blocks = [
        block
        for block in sheet.blocks
        if selected == "all" or block.metadata.get("form") == selected
    ]
    if selected != "all" and len(blocks) != 3:
        raise ProfileBuildError(
            f"Kohlberg MJI: Form {selected} must contain 3 dilemma blocks, found {len(blocks)}."
        )
    label = PROFILE_LABELS["kohlberg_mji_v1"]
    version = sheet.profile_version
    if selected == "all":
        label = f"{label}（全部三套）"
        version = f"{version}-all"
    else:
        label = f"{label}（Form {selected}）"
        version = f"{version}-form-{selected}"
    return replace(
        sheet,
        profile_version=version,
        profile_label=label,
        questions=[slot.question for block in blocks for slot in block.slots],
        blocks=blocks,
        default_shuffle=False,
        shuffle_supported=False,
    )


def _typed_flat_layout(
    spec: TypedFlatSpec,
    cells: Sequence[str],
    language: str,
) -> ProfileLayout:
    """Build typed slots for a conventional A1/A2/A3+ scale workbook."""

    questions = _nonempty(cells[2:])
    expected = len(spec.rules)
    if len(questions) != expected:
        raise ProfileBuildError(
            f"{spec.label}: expected {expected} response items, found {len(questions)}."
        )

    slots: List[ScaleSlot] = []
    for item_no, (question, rule) in enumerate(zip(questions, spec.rules), start=1):
        slot_id = rule.slot_id or f"item_{item_no:03d}"
        slots.append(
            ScaleSlot(
                slot_id=slot_id,
                question=question,
                response_type=rule.response_type,
                minimum=rule.minimum,
                maximum=rule.maximum,
                choices=list(rule.choices),
                context_id="items",
                section_id="items",
                # This is backend metadata, not source material.  Leaving it
                # empty prevents free-mode prompts from injecting a generated
                # section heading between A2 and the original items.
                section_label="",
                metadata=dict(rule.metadata),
            )
        )

    return ProfileLayout(
        profile_id=spec.profile_id,
        profile_version="1",
        profile_label=spec.label,
        instruction=_profile_instruction(cells),
        blocks=[
            ScaleTaskBlock(
                block_id="items",
                label=spec.label,
                slots=slots,
                shuffle_policy=spec.shuffle_policy,
            )
        ],
        default_shuffle=spec.default_shuffle,
        shuffle_supported=spec.shuffle_supported,
    )


def build_profile_layout(
    profile_id: str,
    cells: Sequence[str],
    language: str,
    images: Sequence[ScaleImage],
) -> ProfileLayout:
    """Build and validate a known special-profile layout.

    Failure is explicit.  We do not fall back to a generic flat parser because
    doing so would silently turn context or rating instructions into scoreable
    items.
    """

    typed_flat = TYPED_FLAT_SPECS.get(profile_id)
    if typed_flat is not None:
        return _typed_flat_layout(typed_flat, cells, language)
    if profile_id == "attribution_bias_v1":
        return _attribution_layout(cells, language)
    if profile_id == "human_rights_v1":
        return _human_rights_layout(cells, language)
    if profile_id == "ios_v1":
        return _ios_layout(cells, language, images)
    if profile_id == "intuitive_reasoning_v1":
        return _intuitive_reasoning_layout(cells, language)
    if profile_id == "kohlberg_story_v1":
        return _kohlberg_story_layout(cells, language)
    if profile_id == "kohlberg_mji_v1":
        return _kohlberg_mji_layout(cells, language)
    raise ProfileBuildError(f"Unknown scale profile: {profile_id}")
