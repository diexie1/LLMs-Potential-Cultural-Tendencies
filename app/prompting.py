"""Build source-faithful scale prompts and preserve returned answers."""

from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .scale_loader import ScaleSheet, ScaleSlot
from .provenance import canonical_hash


PARSER_VERSION = "strict-answer-json-v1"
PROFILE_PARSER_VERSION = "strict-slot-json-v2"
FREE_PARSER_VERSION = "free-text-v10"
RESPONSE_STATUSES = (
    "ok",
    "partial",
    "refusal",
    "unparsed",
    "empty_response",
    "transport_error",
)
_NUMERIC_TEXT = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")

# ``strict`` remains accepted as a migration alias for older saved settings,
# but current runs always use source-faithful unconstrained prompts.
PROMPT_CONTRACTS = ("strict", "free", "free_scores_only", "unconstrained")
FREE_PROMPT_CONTRACTS = ("free", "free_scores_only", "unconstrained")
DEFAULT_PROMPT_CONTRACT = "unconstrained"


def normalize_cultural_identity(value: Any) -> str:
    identity = str(value or "none").strip().lower()
    if identity not in {"none", "china", "usa"}:
        raise ValueError("cultural_identity 必须是 none、china 或 usa")
    return identity


def cultural_identity_prompt(identity: Any, language: str) -> str:
    identity = normalize_cultural_identity(identity)
    if identity == "none":
        return ""
    if str(language).casefold().startswith("ch"):
        country = "中国" if identity == "china" else "美国"
        return f"你是一位在{country}出生并生活的普通人，正在回答以下问题。"
    country = "China" if identity == "china" else "the United States"
    return (
        f"You are an average person born and living in {country} "
        "responding to the following questions."
    )


def normalize_prompt_contract(value: Any) -> str:
    contract = str(value or DEFAULT_PROMPT_CONTRACT).strip().lower()
    if contract not in PROMPT_CONTRACTS:
        raise ValueError(
            "prompt_contract 必须是 strict、free、free_scores_only 或 unconstrained"
        )
    if contract in {"free", "strict"}:
        return "unconstrained"
    return contract


def is_free_prompt_contract(value: Any) -> bool:
    """Whether a prompt contract keeps the unconstrained/free parser path."""

    return normalize_prompt_contract(value) in FREE_PROMPT_CONTRACTS


_REFUSAL_PATTERNS = (
    re.compile(r"\bi\s+(?:cannot|can't|can\s*not|am\s+unable\s+to|won't|will\s+not)\s+(?:provide|give|answer|rate|assign|score|offer|complete|respond|express)\b", re.I),
    re.compile(r"\b(?:i\s+)?(?:must\s+decline|prefer\s+not\s+to\s+answer)\b", re.I),
    re.compile(r"(?:我无法|我不能|我不便|拒绝回答|不予回答|不提供|不作答|无法对[^\n。]*(?:评价|判断|打分|评分|给出))"),
    re.compile(r"作为(?:一个)?(?:人工智能|AI)[^。\n]{0,90}不对[^。\n]{0,50}(?:主观评分|价值判断|打分)"),
)


def classify_model_response(raw: Any, parsed: Optional["ParseResult"] = None) -> str:
    """Classify a returned model message without inventing an answer.

    This is deliberately separate from parsing: a refusal or an otherwise
    unparseable response is still a real model response and should be retained
    as a trial result, while transport failures remain retryable errors.
    """

    text = str(raw or "").replace("’", "'").replace("‘", "'")
    if not text.strip():
        return "empty_response"
    if parsed is not None:
        if parsed.status == "ok":
            return "ok"
        if parsed.status == "partial":
            return "partial"
    if any(pattern.search(text) for pattern in _REFUSAL_PATTERNS):
        return "refusal"
    return "unparsed"


# "5", "5.5", "-3"
_STANDALONE_NUMBER = re.compile(r"(?<![\d.])[+-]?\d+(?:\.\d+)?(?![\d.])")
# "5-7", "5 – 7", "2~3", "40 到 60", "5/7"
_RANGE_ALNUM = re.compile(
    r"(?<![\d.])([+-]?\d+(?:\.\d+)?)\s*(?:-|–|—|~|～|至|到|/)\s*([+-]?\d+(?:\.\d+)?)(?![\d.])"
)
# "6or7", "6 or 7", "6或7"
_RANGE_OR = re.compile(
    r"(?<![\d.])([+-]?\d+(?:\.\d+)?)\s*(?:or|或|或者)\s*([+-]?\d+(?:\.\d+)?)(?![\d.])",
    re.IGNORECASE,
)
# "between 3 and 4", "介于 3 和 4"
_RANGE_BETWEEN = re.compile(
    r"(?:between|介于|在)\s*([+-]?\d+(?:\.\d+)?)\s*(?:and|与|和|到|至)\s*([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# In free mode a numeric answer is readable only when it is either the first
# token of the item's answer or follows an explicit score/answer marker. Do
# not search arbitrary prose: numbers in a rationale ("5 people", "2
# reasons", etc.) are not model scores.
_FREE_LEADING_NUMBER = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?![\d.])"
)
_FREE_SCORE_MARKER = re.compile(
    r"(?:\b(?:score|rating|answer|response|numeric\s+answer)\b\s*"
    r"(?:(?:is|=)\s*|:\s*|(?=\d))|"
    r"(?:评分|分数|得分|分值|答案|回答)\s*"
    r"(?:(?:为|是|=|：|:)\s*|(?=\d))|"
    r"我\s*(?:给|打(?:分)?)\s*|"
    r"\bi\s+(?:would\s+)?(?:give|rate)\b"
    r"(?:\s+(?:this|it|a|an|the|score|rating)){0,4}\s*)",
    re.IGNORECASE,
)


@dataclass
class ParsedAnswer:
    """One model response slot, in the order displayed to the model."""

    answer: Any = None
    score: Optional[float] = None
    parse_status: str = "missing"
    parse_error: Optional[str] = None
    slot_id: Optional[str] = None
    mapping_method: Optional[str] = None
    range_text: Optional[str] = None
    range_lower: Optional[float] = None
    range_upper: Optional[float] = None
    range_unit: Optional[str] = None


@dataclass
class ParseResult:
    answers: List[ParsedAnswer]
    status: str
    error: Optional[str] = None
    recovery: Optional[str] = None


@dataclass
class DisplayBlock:
    """One prompt block after profile-aware ordering has been applied."""

    block_id: str
    label: str
    context: str
    instruction: str
    slots: List[ScaleSlot]
    original_indices: List[int]
    shuffle_policy: str


@dataclass
class DisplayPlan:
    """An auditable display-to-canonical mapping for a structured profile."""

    blocks: List[DisplayBlock]
    original_indices: List[int]
    shuffle_enabled: bool

    @property
    def slots(self) -> List[ScaleSlot]:
        return [slot for block in self.blocks for slot in block.slots]

    @property
    def n_items(self) -> int:
        return len(self.original_indices)

    def metadata(self) -> Dict[str, Any]:
        display_index = 0
        blocks: List[Dict[str, Any]] = []
        for block_index, block in enumerate(self.blocks, start=1):
            slot_rows = []
            for slot, original_index in zip(block.slots, block.original_indices):
                display_index += 1
                slot_rows.append(
                    {
                        "display_index": display_index,
                        "slot_id": slot.slot_id,
                        "original_index": original_index + 1,
                    }
                )
            blocks.append(
                {
                    "display_block_index": block_index,
                    "block_id": block.block_id,
                    "shuffle_policy": block.shuffle_policy,
                    "slots": slot_rows,
                }
            )
        return {
            "version": "profile-display-plan-v1",
            "shuffle_enabled": self.shuffle_enabled,
            "blocks": blocks,
        }


def build_display_plan(
    sheet: ScaleSheet,
    shuffle_enabled: bool,
    rng: Optional[random.Random] = None,
) -> DisplayPlan:
    """Create a shuffle plan without separating a scenario from its slots.

    ``block`` moves whole contextual scenarios, ``within`` permutes only slots
    inside a block, and ``fixed`` remains unchanged.  This is intentionally
    more conservative than a flat question shuffle.
    """

    if not sheet.is_profiled:
        raise ValueError("A profile display plan requires a profiled ScaleSheet.")
    randomizer = rng or random.Random()
    enabled = bool(shuffle_enabled and sheet.shuffle_supported)
    canonical_slots = sheet.slots
    index_by_slot_id = {slot.slot_id: index for index, slot in enumerate(canonical_slots)}
    if len(index_by_slot_id) != len(canonical_slots):
        raise ValueError("Profile slot IDs must be unique.")

    source_blocks = list(sheet.blocks)
    selected_blocks = list(source_blocks)
    if enabled:
        movable_positions = [
            index
            for index, block in enumerate(source_blocks)
            if block.shuffle_policy == "block"
        ]
        movable_blocks = [source_blocks[index] for index in movable_positions]
        randomizer.shuffle(movable_blocks)
        for position, block in zip(movable_positions, movable_blocks):
            selected_blocks[position] = block

    display_blocks: List[DisplayBlock] = []
    flat_indices: List[int] = []
    for block in selected_blocks:
        slots = list(block.slots)
        if enabled and block.shuffle_policy == "within":
            randomizer.shuffle(slots)
        original_indices = [index_by_slot_id[slot.slot_id] for slot in slots]
        flat_indices.extend(original_indices)
        display_blocks.append(
            DisplayBlock(
                block_id=block.block_id,
                label=block.label,
                context=block.context,
                instruction=block.instruction,
                slots=slots,
                original_indices=original_indices,
                shuffle_policy=block.shuffle_policy,
            )
        )
    return DisplayPlan(
        blocks=display_blocks,
        original_indices=flat_indices,
        shuffle_enabled=enabled,
    )


def _source_section_line(text: str, marker: str) -> str:
    """Keep a source section marker while tolerating parser-normalized text."""

    body = str(text or "").strip()
    if not body:
        return f"{marker}."
    if re.match(rf"^{re.escape(marker)}[.．]\s*", body, re.IGNORECASE):
        return body
    return f"{marker}. {body}"


def _free_scores_only_instruction(language: str) -> str:
    """One optional line added to the scale guidance as an experimental condition."""

    if language.casefold().startswith("ch"):
        return "请仅给出题目要求的分数，不需要解释。"
    return "Please give only the requested scores, without explanations."


def _build_attribution_free_prompt(
    sheet: ScaleSheet,
    language: str,
    plan: DisplayPlan,
    *,
    scores_only: bool = False,
) -> str:
    """Render attribution bias as source-faithful, shuffled scenario blocks.

    The six scenarios are shuffled only as complete blocks by
    :func:`build_display_plan`. The groupness task is a separate fixed block
    rendered once after the scenarios. A1 is deliberately omitted for this
    Lu-style free prompt; A2 and all source task instructions remain visible.
    """

    parts: List[str] = []
    if sheet.instruction.strip():
        parts.append(sheet.instruction.strip())
    if scores_only:
        parts.append(_free_scores_only_instruction(language))

    for block in plan.blocks:
        block_lines: List[str] = []
        if block.context.strip():
            block_lines.append(block.context.strip())

        if block.block_id == "attribution_groupness":
            # A9 has its own response scale. It is not the instruction for
            # scenario six and must not be repeated once per item.
            if block.instruction.strip():
                block_lines.append(block.instruction.strip())
            for slot in block.slots:
                block_lines.append(slot.question.strip())
        else:
            a_slots = [
                slot for slot in block.slots
                if slot.metadata.get("subtask") == "attribution_rating"
            ]
            b_slots = [
                slot for slot in block.slots
                if slot.metadata.get("subtask") == "other_actor_probability"
            ]
            c_slots = [
                slot for slot in block.slots
                if slot.metadata.get("subtask") == "same_actor_probability"
            ]

            if a_slots:
                block_lines.append(_source_section_line(a_slots[0].section_label, "A"))
                for slot in a_slots:
                    block_lines.append(slot.question.strip())
            for slot in b_slots:
                block_lines.append(
                    _source_section_line(slot.question, "B")
                )
            for slot in c_slots:
                block_lines.append(
                    _source_section_line(slot.question, "C")
                )

        if block_lines:
            parts.append("\n".join(block_lines))

    return "\n\n".join(parts)


def _build_free_prompt(
    sheet: ScaleSheet,
    language: str,
    ordered_slots: Sequence[ScaleSlot],
    display_plan: Optional[DisplayPlan] = None,
    *,
    scores_only: bool = False,
) -> str:
    """Render the unconstrained (Lu et al. 2025 main-analysis) prompt.

    Instruction, source context and question text only. Generated slot labels,
    answer-type hints, output formats and answer-count requirements are absent.
    """

    if sheet.profile_id == "attribution_bias_v1" and display_plan is not None:
        return _build_attribution_free_prompt(
            sheet, language, display_plan, scores_only=scores_only
        )

    parts: List[str] = []
    # A1 is workbook metadata, not task content. Do not send the scale title
    # in the free prompt; keep it in the result payload for identification and
    # provenance.
    if sheet.instruction.strip():
        parts.append(sheet.instruction.strip())
    if scores_only:
        parts.append(_free_scores_only_instruction(language))

    # Context blocks keep the task coherent; they are scenario material, not
    # answer slots, and Lu-style prompts carry them too. Build the slot-to-
    # context map once — slot identity is preserved from sheet.blocks.
    context_by_slot: Dict[int, str] = {}
    section_by_slot: Dict[int, str] = {}
    for block in sheet.blocks:
        context = block.context.strip()
        extra = block.instruction.strip()
        if extra and extra != sheet.instruction.strip():
            context = (context + "\n" + extra).strip()
        for slot in block.slots:
            context_by_slot[id(slot)] = context
            # Profile parsers keep the source section heading/instruction on
            # each slot. In free mode it is source task content and must stay
            # in the prompt; only backend-only labels such as slot_id remain
            # suppressed.
            section_by_slot[id(slot)] = slot.section_label.strip()

    lines: List[str] = []
    previous_context: Optional[str] = None
    previous_section: Optional[str] = None
    for slot in ordered_slots:
        context = context_by_slot.get(id(slot), "")
        if context and context != previous_context:
            if lines:
                lines.append("")
            lines.append(context)
            previous_context = context
            previous_section = None
        section = section_by_slot.get(id(slot), "")
        if section and section != previous_section:
            lines.append(section)
            previous_section = section
        lines.append(slot.question.strip())

    parts.append("\n".join(lines))
    return "\n\n".join(part for part in parts if part)


def _build_base_prompt(
    sheet: ScaleSheet,
    order: Union[Sequence[int], DisplayPlan],
    language: str = "",
    contract: str = DEFAULT_PROMPT_CONTRACT,
) -> str:
    """Render source content, with only the optional score instruction."""

    contract = normalize_prompt_contract(contract)
    lang = (language or sheet.language or "").casefold()
    plan = None
    if sheet.is_profiled:
        plan = order if isinstance(order, DisplayPlan) else build_display_plan(sheet, False)
        ordered = plan.slots
    else:
        slots = sheet.slots
        ordered = [slots[index] for index in order]
    return _build_free_prompt(
        sheet, lang, ordered, display_plan=plan,
        scores_only=contract == "free_scores_only",
    )


def build_prompt(
    sheet: ScaleSheet,
    order: Union[Sequence[int], DisplayPlan],
    language: str = "",
    contract: str = DEFAULT_PROMPT_CONTRACT,
    *,
    cultural_identity: str = "none",
) -> str:
    """Prepend the identity condition once, retaining all source task content."""

    base = _build_base_prompt(sheet, order, language, contract)
    prefix = cultural_identity_prompt(cultural_identity, language or sheet.language)
    return f"{prefix}\n\n{base}" if prefix else base


def build_trial_prompt(
    sheet: ScaleSheet,
    language: str,
    shuffle_enabled: bool,
    rng: Optional[random.Random],
    contract: str = DEFAULT_PROMPT_CONTRACT,
    cultural_identity: str = "none",
) -> Tuple[str, List[int], Optional[DisplayPlan]]:
    """Use the same ordering and text renderer for preview and model calls."""

    plan = build_display_plan(sheet, shuffle_enabled, rng) if sheet.is_profiled else None
    order = plan.original_indices if plan else (
        shuffle_order(sheet.n_items, rng) if shuffle_enabled else list(range(sheet.n_items))
    )
    return (
        build_prompt(sheet, plan or order, language, contract, cultural_identity=cultural_identity),
        order, plan,
    )


def prompt_preview_sections(sheet: ScaleSheet, language: str, prompt: str,
                            contract: str, cultural_identity: str) -> List[Dict[str, str]]:
    """Annotate exact prompt segments for display; labels never enter the text."""

    sections: List[Dict[str, str]] = []
    remaining = prompt
    prefixes = [
        ("平台追加：身份提示", cultural_identity_prompt(cultural_identity, language)),
        ("原量表指导语", sheet.instruction.strip()),
        ("平台追加：只答分数提示", _free_scores_only_instruction(language)
         if normalize_prompt_contract(contract) == "free_scores_only" else ""),
    ]
    for label, text in prefixes:
        if text and remaining.startswith(text):
            sections.append({"label": label, "text": text})
            remaining = remaining[len(text):]
            if remaining.startswith("\n\n"):
                remaining = remaining[2:]
    if remaining:
        sections.append({"label": "量表题目与情境原文", "text": remaining})
    return sections


def _failed_parse(n_items: int, status: str, error: str) -> ParseResult:
    return ParseResult(
        answers=[
            ParsedAnswer(parse_status=status, parse_error=error)
            for _ in range(n_items)
        ],
        status=status,
        error=error,
    )


def _load_strict_json(text: str) -> Any:
    """Parse either a bare JSON object or a JSON-only fenced code block.

    Deliberately do not search for a JSON fragment within prose. A fragment search would
    reintroduce the old failure mode where explanation/scenario numbers become scores.
    """

    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", stripped, re.IGNORECASE)
    candidate = fenced.group(1) if fenced else stripped
    return json.loads(candidate)


def _display_index(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _numeric_score(value: Any) -> Optional[float]:
    """Accept only an explicitly supplied finite scalar, never a number in prose."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            score = float(value)
        elif isinstance(value, str) and _NUMERIC_TEXT.fullmatch(value.strip()):
            score = float(value.strip())
        else:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return score if math.isfinite(score) else None


def _coerce_flat_entries(payload: Any, n_items: int) -> Optional[List[Dict[str, Any]]]:
    """Normalize the outer envelope of a flat (non-profiled) response.

    Accepts the canonical ``{"answers": [...]}`` plus two equivalent shapes the
    model sometimes emits: a bare answers array ``[...]`` and a map keyed by
    display index (``{1: v}`` / ``{"1": v}``).  Returns ``None`` when the shape
    is not recognized, so validation stays conservative -- a missing score is
    preferred over a value guessed from prose.
    """

    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    entries = payload.get("answers")
    if isinstance(entries, list):
        return entries
    out: List[Dict[str, Any]] = []
    for key, value in payload.items():
        index = _display_index(key)
        if index is None or not (1 <= index <= n_items):
            return None
        out.append({"display_index": index, "answer": value})
    return out


def _coerce_slot_entries(payload: Any, by_id: Dict[str, int]) -> Optional[List[Dict[str, Any]]]:
    """Normalize the outer envelope of a profiled (slot-id) response.

    Accepts the canonical ``{"answers": [{"slot_id", "answer"}, ...]}`` plus the
    map shape the model sometimes emits instead: ``{slot_id: {"slot_id", "answer"}}``
    or ``{slot_id: value}``.  Only converts when *every* key is a declared slot id,
    so a stray or prose-filled object is still rejected rather than guessed.
    """

    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    entries = payload.get("answers")
    if isinstance(entries, list):
        return entries
    out: List[Dict[str, Any]] = []
    for key, value in payload.items():
        if not isinstance(key, str) or key not in by_id:
            return None
        if isinstance(value, dict) and "answer" in value:
            out.append({"slot_id": key, "answer": value["answer"]})
        else:
            out.append({"slot_id": key, "answer": value})
    return out


def _recover_slot_jsonl(text: str, valid_slot_ids: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Safely recover one JSON object per line from a profiled response.

    Some models ignore the outer ``answers`` envelope and emit JSONL instead.
    Recovery is intentionally narrow: every non-empty line must be an object
    with exactly ``slot_id`` and ``answer``, and the set of IDs must match the
    declared profile exactly.  We never extract JSON fragments from prose.
    """

    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    expected = set(valid_slot_ids)
    if not lines or len(lines) != len(expected):
        return None
    entries: List[Dict[str, Any]] = []
    seen = set()
    for line in lines:
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(value, dict) or set(value) != {"slot_id", "answer"}:
            return None
        slot_id = value.get("slot_id")
        if not isinstance(slot_id, str) or slot_id not in expected or slot_id in seen:
            return None
        seen.add(slot_id)
        entries.append(value)
    if seen != expected:
        return None
    return {"answers": entries}


def parse_answers(text: str, n_items: int) -> ParseResult:
    """Parse the declared response schema without any positional-number fallback.

    A malformed response is retained in ``raw_response`` by the runner, while all scores
    remain null. This is intentionally conservative: missing data is preferable to a
    fabricated score derived from payoff matrices, instructions, or explanations.
    """

    try:
        payload = _load_strict_json(text)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return _failed_parse(n_items, "invalid_json", str(exc))

    entries = _coerce_flat_entries(payload, n_items)
    if entries is None:
        reason = (
            "Top-level JSON must be an object."
            if isinstance(payload, list)
            else "JSON must contain an answers list."
        )
        return _failed_parse(n_items, "invalid_schema", reason)

    answers = [ParsedAnswer() for _ in range(n_items)]
    seen = set()
    protocol_errors: List[str] = []

    for entry in entries:
        if not isinstance(entry, dict):
            protocol_errors.append("An answers entry is not an object.")
            continue
        display_index = _display_index(entry.get("display_index"))
        if display_index is None or not (1 <= display_index <= n_items):
            protocol_errors.append("An answers entry has an invalid display_index.")
            continue
        slot = display_index - 1
        if slot in seen:
            answers[slot] = ParsedAnswer(
                parse_status="duplicate_display_index",
                parse_error="Each display_index may occur only once.",
            )
            protocol_errors.append(f"Duplicate display_index {display_index}.")
            continue
        seen.add(slot)

        if "answer" not in entry:
            answers[slot] = ParsedAnswer(
                parse_status="missing_answer",
                parse_error="The response slot has no answer field.",
            )
            protocol_errors.append(f"Missing answer for display_index {display_index}.")
            continue

        value = entry["answer"]
        if value is None:
            answers[slot] = ParsedAnswer(
                answer=None,
                parse_status="null_answer",
                parse_error="null is not a valid final answer.",
            )
            protocol_errors.append(f"Null answer for display_index {display_index}.")
        elif isinstance(value, str) and not value.strip():
            answers[slot] = ParsedAnswer(
                answer=value,
                parse_status="empty_answer",
                parse_error="answer must not be an empty string.",
            )
            protocol_errors.append(f"Empty answer for display_index {display_index}.")
        elif isinstance(value, bool) or not isinstance(value, (str, int, float)):
            answers[slot] = ParsedAnswer(
                answer=value,
                parse_status="invalid_answer_type",
                parse_error="answer must be a string, finite number, or null.",
            )
            protocol_errors.append(f"Invalid answer type for display_index {display_index}.")
        else:
            numeric = _numeric_score(value)
            numeric_candidate = isinstance(value, (int, float)) or (
                isinstance(value, str) and bool(_NUMERIC_TEXT.fullmatch(value.strip()))
            )
            if numeric_candidate and numeric is None:
                answers[slot] = ParsedAnswer(
                    answer=value,
                    parse_status="invalid_numeric_answer",
                    parse_error="Numeric answer must be finite.",
                )
                protocol_errors.append(f"Non-finite numeric answer for display_index {display_index}.")
            else:
                parse_status = "parsed"
                parse_error = None
                if isinstance(value, str) and numeric is None:
                    _, uncertainty_status = _free_extract_numeric(value)
                    if uncertainty_status == "range_unresolved":
                        parse_status = uncertainty_status
                        parse_error = (
                            "The response gives multiple possible scores; "
                            "no single score was guessed."
                        )
                        protocol_errors.append(
                            f"Unresolved score range for display_index {display_index}."
                        )
                answers[slot] = ParsedAnswer(
                    answer=value,
                    score=numeric,
                    parse_status=parse_status,
                    parse_error=parse_error,
                )

    missing = [i + 1 for i, answer in enumerate(answers) if answer.parse_status == "missing"]
    if missing:
        protocol_errors.append("Missing display_index: " + ", ".join(map(str, missing)) + ".")

    if protocol_errors or missing:
        return ParseResult(
            answers=answers,
            status="partial",
            error=" ".join(protocol_errors) if protocol_errors else None,
        )
    return ParseResult(answers=answers, status="ok")


def _failed_profile_parse(plan: DisplayPlan, status: str, error: str) -> ParseResult:
    return ParseResult(
        answers=[
            ParsedAnswer(slot_id=slot.slot_id, parse_status=status, parse_error=error)
            for slot in plan.slots
        ],
        status=status,
        error=error,
    )


def _normalise_choice(value: Any, choices: Sequence[str]) -> Optional[str]:
    """Validate a categorical answer without extracting values from prose."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = _numeric_score(value)
        if numeric is None:
            return None
        text = str(int(numeric)) if float(numeric).is_integer() else str(numeric)
    elif isinstance(value, str):
        text = value.strip()
    else:
        return None
    if not text:
        return None
    canonical = {choice.casefold(): choice for choice in choices}
    direct = canonical.get(text.casefold())
    if direct is not None:
        return direct
    if {choice.casefold() for choice in choices} == {"yes", "no"}:
        aliases = {
            "y": "YES",
            "yes": "YES",
            "是": "YES",
            "正确": "YES",
            "n": "NO",
            "no": "NO",
            "否": "NO",
            "不": "NO",
            "错误": "NO",
        }
        answer = aliases.get(text.casefold())
        if answer is not None:
            return canonical.get(answer.casefold())
    return None


_PAIR_CODE_PATTERN = re.compile(
    r"(?<!\d)(?:(?:circle|圆圈)\s*)?(?:第\s*)?([1-7])\s*(?:个\s*)?"
    r"(?:[-–—/／&,]|and|和|与|及|、)\s*"
    r"(?:(?:circle|圆圈)\s*)?(?:第\s*)?([1-7])(?!\d)",
    re.IGNORECASE,
)


def _normalise_pair_code(value: Any, choices: Sequence[str]) -> Optional[str]:
    """Normalize an IOS two-circle code such as ``3-4``.

    The code is categorical: it must never be sent through numeric range
    parsing or converted to a midpoint.
    """

    if value is None or isinstance(value, bool):
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return None
    match = _PAIR_CODE_PATTERN.search(text)
    if not match:
        return None
    left, right = int(match.group(1)), int(match.group(2))
    if left == right:
        return None
    canonical = f"{min(left, right)}-{max(left, right)}"
    allowed = {str(choice).strip() for choice in choices}
    return canonical if canonical in allowed else None


def _free_extract_pair_code(body: str, choices: Sequence[str]) -> Optional[str]:
    """Read an IOS circle-pair code from a short free-form answer."""

    text = unicodedata.normalize("NFKC", str(body or ""))
    direct = _normalise_pair_code(text.strip(), choices)
    if direct is not None:
        return direct
    marker = re.search(
        r"(?:answer|response|choice|selected|选择|答案|回答|选的是?)"
        r"\s*[:：=]?\s*(.*)",
        text,
        re.IGNORECASE,
    )
    if marker:
        marked = _normalise_pair_code(marker.group(1), choices)
        if marked is not None:
            return marked
    # The pair code normally appears at the start of the answer line. Avoid
    # searching a long explanation where unrelated numbers may occur.
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if len(first_line) <= 120:
        return _normalise_pair_code(first_line, choices)
    return None


def _normalise_ios_pair(value: Any, choices: Sequence[str]) -> Optional[str]:
    """Normalize one IOS *whole-diagram* label (1–7).

    IOS is sometimes described informally as a pair choice, but the answer is
    the label of one pre-drawn pair of circles.  Keep this separate from the
    legacy ``pair_code`` parser, which accepts two circle numbers such as
    ``3-4``.
    """

    if value is None or isinstance(value, bool):
        return None
    direct = _normalise_choice(value, choices)
    if direct is not None:
        return direct
    return _normalise_choice(_free_choice_token(value), choices)


def _parse_profile_value(slot: ScaleSlot, value: Any) -> ParsedAnswer:
    """Validate one response against its declared slot type and range."""

    base = {"slot_id": slot.slot_id}
    if value is None:
        return ParsedAnswer(
            **base,
            parse_status="null_answer",
            parse_error="null is not a valid final answer.",
        )
    if isinstance(value, str) and not value.strip():
        return ParsedAnswer(
            **base,
            answer=value,
            parse_status="empty_answer",
            parse_error="answer must not be an empty string.",
        )
    if slot.response_type == "ios_pair":
        answer = _normalise_ios_pair(value, slot.choices)
        if answer is None:
            return ParsedAnswer(
                **base,
                answer=value,
                parse_status="invalid_ios_pair",
                parse_error=f"answer must be one IOS pair-diagram label from: {', '.join(slot.choices)}.",
            )
        return ParsedAnswer(
            **base,
            answer=answer,
            score=_numeric_score(answer),
            parse_status="parsed",
        )
    if slot.response_type == "pair_code":
        answer = _normalise_pair_code(value, slot.choices)
        if answer is None:
            return ParsedAnswer(
                **base,
                answer=value,
                parse_status="invalid_pair_code",
                parse_error=f"answer must be one valid circle-pair code from: {', '.join(slot.choices)}.",
            )
        return ParsedAnswer(
            **base,
            answer=answer,
            score=None,
            parse_status="parsed",
        )
    if slot.response_type == "choice":
        answer = _normalise_choice(value, slot.choices)
        if answer is None:
            return ParsedAnswer(
                **base,
                answer=value,
                parse_status="invalid_choice",
                parse_error=f"answer must be one of: {', '.join(slot.choices)}.",
            )
        return ParsedAnswer(
            **base,
            answer=answer,
            score=_numeric_score(answer),
            parse_status="parsed",
        )
    if slot.response_type == "integer":
        numeric = _numeric_score(value)
        if numeric is None or not float(numeric).is_integer():
            return ParsedAnswer(
                **base,
                answer=value,
                parse_status="invalid_integer",
                parse_error="answer must be a finite integer.",
            )
        if slot.minimum is not None and numeric < slot.minimum:
            return ParsedAnswer(
                **base,
                answer=value,
                parse_status="below_minimum",
                parse_error=f"answer must be at least {slot.minimum}.",
            )
        if slot.maximum is not None and numeric > slot.maximum:
            return ParsedAnswer(
                **base,
                answer=value,
                parse_status="above_maximum",
                parse_error=f"answer must be at most {slot.maximum}.",
            )
        integer = int(numeric)
        return ParsedAnswer(
            **base,
            answer=integer,
            score=float(integer),
            parse_status="parsed",
        )
    if slot.response_type == "number":
        numeric = _numeric_score(value)
        if numeric is None:
            return ParsedAnswer(
                **base,
                answer=value,
                parse_status="invalid_numeric_answer",
                parse_error="answer must be one finite numeric value.",
            )
        if slot.minimum is not None and numeric < slot.minimum:
            return ParsedAnswer(
                **base,
                answer=value,
                parse_status="below_minimum",
                parse_error=f"answer must be at least {slot.minimum}.",
            )
        if slot.maximum is not None and numeric > slot.maximum:
            return ParsedAnswer(
                **base,
                answer=value,
                parse_status="above_maximum",
                parse_error=f"answer must be at most {slot.maximum}.",
            )
        return ParsedAnswer(
            **base,
            answer=value,
            score=numeric,
            parse_status="parsed",
        )
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return ParsedAnswer(
            **base,
            answer=value,
            parse_status="invalid_answer_type",
            parse_error="answer must be a string or finite number.",
        )
    numeric = _numeric_score(value)
    if isinstance(value, (int, float)) and numeric is None:
        return ParsedAnswer(
            **base,
            answer=value,
            parse_status="invalid_numeric_answer",
            parse_error="numeric answer must be finite.",
        )
    return ParsedAnswer(**base, answer=value, score=numeric, parse_status="parsed")


def parse_profile_answers(text: str, plan: DisplayPlan) -> ParseResult:
    """Strictly parse responses keyed by stable profile slot IDs.

    Unlike the legacy flat parser, this parser has no display-number fallback.
    An answer can only be attached to a declared slot ID and must meet that
    slot's response type and range.  Invalid data remains invalid rather than
    being guessed from surrounding explanatory text.
    """

    slots = plan.slots
    by_id = {slot.slot_id: (index, slot) for index, slot in enumerate(slots)}
    recovery: Optional[str] = None
    try:
        payload = _load_strict_json(text)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        payload = _recover_slot_jsonl(text, list(by_id))
        if payload is None:
            return _failed_profile_parse(plan, "invalid_json", str(exc))
        recovery = "slot_jsonl_v1"
    entries = _coerce_slot_entries(payload, by_id)
    if entries is None:
        reason = (
            "Top-level JSON must be an object."
            if isinstance(payload, list)
            else "JSON must contain an answers list."
        )
        return _failed_profile_parse(plan, "invalid_schema", reason)
    answers = [ParsedAnswer(slot_id=slot.slot_id) for slot in slots]
    seen = set()
    protocol_errors: List[str] = []

    for entry in entries:
        if not isinstance(entry, dict):
            protocol_errors.append("An answers entry is not an object.")
            continue
        slot_id = entry.get("slot_id")
        if not isinstance(slot_id, str) or slot_id not in by_id:
            protocol_errors.append("An answers entry has an unknown or invalid slot_id.")
            continue
        position, slot = by_id[slot_id]
        if slot_id in seen:
            answers[position] = ParsedAnswer(
                slot_id=slot_id,
                parse_status="duplicate_slot_id",
                parse_error="Each slot_id may occur only once.",
            )
            protocol_errors.append(f"Duplicate slot_id {slot_id}.")
            continue
        seen.add(slot_id)
        if "answer" not in entry:
            answers[position] = ParsedAnswer(
                slot_id=slot_id,
                parse_status="missing_answer",
                parse_error="The response slot has no answer field.",
            )
            protocol_errors.append(f"Missing answer for slot_id {slot_id}.")
            continue
        parsed = _parse_profile_value(slot, entry["answer"])
        answers[position] = parsed
        if parsed.parse_status != "parsed":
            protocol_errors.append(f"Invalid answer for slot_id {slot_id}: {parsed.parse_error}")

    missing = [slot.slot_id for slot, answer in zip(slots, answers) if answer.parse_status == "missing"]
    if missing:
        protocol_errors.append("Missing slot_id: " + ", ".join(missing) + ".")
    if protocol_errors:
        return ParseResult(
            answers=answers,
            status="partial",
            error=" ".join(protocol_errors),
            recovery=recovery,
        )
    return ParseResult(answers=answers, status="ok", recovery=recovery)


# A line that opens with an item number: "1. ...", "2、...", "3) ...", "4：..."
_FREE_ITEM_LINE = re.compile(r"^\s*[（(]?\s*(\d+)\s*[.、．)\]）:：]\s*(.*)$")
_FREE_ITEM_LABEL = re.compile(
    r"^\s*(?:第\s*)?(\d+)\s*(?:题|问|item|question|q)?\s*[:：]\s*(.*)$",
    re.IGNORECASE,
)
# Kohlberg stories contain their own numbered follow-up questions.  Only an
# explicit story/dilemma label is a safe outer boundary; bare ``1.`` / ``2.``
# lines are intentionally not used by the dedicated MJI decoder.
_KOHLBERG_STORY_MARKER = re.compile(
    r"^\s*(?:[*_`~]\s*)*"
    r"(?:story|dilemma|情境|故事)\s*(?:#|编号|号)?\s*"
    r"(\d+|[一二三四五六七八九])\s*"
    r"(?:[*_`~]\s*)*(?:[:：.、)）-]\s*|$)(.*)$",
    re.IGNORECASE,
)
# A Markdown table row: "| 1 | 5 |" or "| 1 | 5 | 说明 |". Models reach for a
# table on their own often enough that ignoring it would discard valid scores.
_FREE_TABLE_ROW = re.compile(r"^\s*\|\s*(\d+)\s*\|\s*([^|]*)\|")
_FREE_TABLE_DIVIDER = re.compile(r"^\s*:?-{3,}:?\s*$")
_FREE_TABLE_ITEM_LABEL = re.compile(
    r"^\s*(\d+)\s*(?:[.)、．。:：）)]\s*)(.+?)\s*$"
)
_FREE_ANSWER_HEADER_HINTS = (
    "answer", "response", "score", "rating", "choice", "selection",
    "答案", "作答", "回答", "评分", "分数", "得分", "选择", "判断", "结果",
)
_FREE_QUESTION_HEADER_HINTS = (
    "question", "item", "statement", "prompt", "题目", "题干", "选项", "内容",
)

# Ranges and open text are readable answers without a single numeric score.
FREE_STATUS_SCORED = ("parsed", "out_of_range")
FREE_STATUS_READABLE = ("parsed", "out_of_range", "text_response", "range_response", "range_out_of_range")
FREE_STATUS_UNSCORED = ("unparsed", "missing", "range_unresolved", "range_response", "range_out_of_range")


def _free_table_item_number(value: Any) -> Optional[int]:
    """Read a table item number without mistaking a decimal for item 1."""

    first = unicodedata.normalize("NFKC", str(value or "")).strip()
    if first.isdigit():
        return int(first)
    match = _FREE_TABLE_ITEM_LABEL.match(first)
    if match is None:
        return None
    remainder = match.group(2).strip()
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", remainder):
        return None
    return int(match.group(1))


def _is_open_text_slot(slot: ScaleSlot) -> bool:
    """Only explicitly open-ended tasks bypass numeric/choice decoding."""

    return slot.response_type == "text" and bool(slot.metadata.get("open_ended"))


def _free_text_response(slot: ScaleSlot, value: str) -> ParsedAnswer:
    """Keep non-numeric answers readable for generic text slots."""

    return ParsedAnswer(
        answer=value,
        score=None,
        slot_id=slot.slot_id,
        parse_status="text_response",
    )


def _unlabeled_text_blocks(text: str, expected: int) -> Optional[List[str]]:
    """Split free prose only when blank-line blocks map one-to-one to text items."""

    blocks = [part.strip() for part in re.split(r"\n\s*\n+", str(text or "").strip()) if part.strip()]
    return blocks if len(blocks) == expected else None


def _free_extract_interval(value: str) -> Optional[Tuple[str, float, float, str]]:
    """Read a closed interval at an answer's start, never discrete alternatives.

    A slash is excluded: ``2/7`` can denote a scale score. Percent endpoints
    retain their original numeric scale (40%, not 0.4).
    """
    target = unicodedata.normalize("NFKC", str(value)).replace("−", "-")
    target = re.sub(r"[*_`]", "", target).strip()
    target = _FREE_SCORE_MARKER.sub("", target, count=1) if _FREE_SCORE_MARKER.match(target) else target
    target = re.sub(r"^(?:约|大约|about|approximately|around)\s*", "", target, flags=re.I)
    number = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
    lower = rf"(?P<lower>{number})\s*(?P<unit1>%|percent\b)?\s*"
    upper = rf"(?P<upper>{number})(?![\d.])\s*(?P<unit2>%|percent\b)?"
    patterns = (
        rf"^{lower}(?:[–—~～-]|至|到|\bto\b)\s*{upper}",
        rf"^(?:between|介于|在)\s*{lower}(?:and|与|和|到|至)\s*{upper}",
        rf"^from\s+{lower}to\s*{upper}",
        rf"^\[\s*{lower},\s*{upper}\s*\]",
    )
    for pattern in patterns:
        match = re.match(pattern, target, re.I)
        if match:
            # "3–5 or 6–7" supplies alternatives, not one interval.
            if re.match(r"^\s*(?:分|人|points?\b|people\b)?\s*"
                        r"(?:or\b|and\b|或|或者|和|与|/|[,;，；、]|[–—~～-]|至|到|to\b)"
                        r"\s*[\[(]?\s*[+-]?\d", target[match.end():], re.I):
                return None
            return (match.group(0).strip(), float(match.group("lower")),
                    float(match.group("upper")), "%" if match.group("unit1") or match.group("unit2") else "")
    return None


def _free_range_answer(slot: ScaleSlot, body: Any, interval: Tuple[str, float, float, str]) -> ParsedAnswer:
    text, lower, upper, unit = interval
    answer = ParsedAnswer(answer=body, slot_id=slot.slot_id, parse_status="range_response",
                          range_text=text, range_lower=lower, range_upper=upper, range_unit=unit)
    if lower > upper:
        answer.parse_status = "range_unresolved"
        answer.parse_error = "The interval endpoints are reversed; they were not swapped."
    elif ((slot.minimum is not None and lower < slot.minimum)
          or (slot.maximum is not None and upper > slot.maximum)):
        answer.parse_status = "range_out_of_range"
        answer.parse_error = "The supplied interval exceeds the source bounds; it was not clipped."
    elif slot.choices:
        allowed = {_numeric_score(choice) for choice in slot.choices}
        if (lower not in allowed or upper not in allowed
            or (all(x is not None and float(x).is_integer() for x in allowed)
                and any(x not in allowed for x in range(int(lower), int(upper) + 1)))):
            answer.parse_status = "invalid_choice"
            answer.parse_error = "The supplied interval includes values outside the source choices; it was retained."
    return answer


def _free_extract_numeric(body: str) -> tuple[Optional[float], str]:
    """Read one score out of an unconstrained answer body.

    A range or alternative is deliberately *not* collapsed to a midpoint:
    the model did not provide one determinate score. Returns ``(None,
    "range_unresolved")`` while the caller preserves the original answer text.
    A score is never invented from an uncertain answer.
    """

    body = unicodedata.normalize("NFKC", str(body or "")).strip()

    def read_target(target: str) -> tuple[Optional[float], str]:
        target = target.lstrip(" \t:：=")
        # Models often wrap an answer in Markdown emphasis, especially when
        # they answer a numbered item as ``**4 = Agree**``.  Removing only
        # leading markup keeps the conservative leading-value rule intact.
        target = re.sub(r"^(?:[*_`~]\s*)+", "", target)
        if (
            _RANGE_ALNUM.match(target)
            or _RANGE_OR.match(target)
            or _RANGE_BETWEEN.match(target)
            or re.match(r"^[+-]?\d+(?:\.\d+)?\s*%\s*"
                        r"(?:[—–~/-]|至|到|or\b|and\b|或|或者|和|与)\s*[+-]?\d", target, re.I)
        ):
            return None, "range_unresolved"
        match = _FREE_LEADING_NUMBER.match(target)
        if match:
            return float(match.group(1)), "parsed"
        return None, "unparsed"

    # A leading scalar is the normal answer form, e.g. ``5 —— reason``. A
    # leading range is checked first so ``5 or 6`` is never reduced to 5.
    numeric, status = read_target(body)
    if status != "unparsed":
        return numeric, status

    # Also accept an explicit label such as ``评分：5`` or ``score = 5``.
    # The marker is deliberately bounded to the same line and a short span;
    # it cannot jump to an unrelated number later in a long explanation.
    marker = _FREE_SCORE_MARKER.search(body)
    if marker:
        numeric, status = read_target(body[marker.end():])
        if status != "unparsed":
            return numeric, status

    # A common spontaneous format is ``question text：3`` or
    # ``question text：**4 = Agree**``.  Restrict this recovery to the first
    # non-empty answer line and require the suffix to begin with the candidate
    # value; arbitrary numbers in a rationale are never scanned.
    first_line = next((line.strip() for line in body.splitlines() if line.strip()), "")
    separator = re.search(r"[:：=]", first_line)
    if separator:
        suffix = first_line[separator.end():].strip()
        numeric, status = read_target(suffix)
        if status == "range_unresolved":
            return numeric, status
        if numeric is not None:
            cleaned_suffix = re.sub(r"^(?:[*_`~]\s*)+", "", suffix).strip()
            leading = _FREE_LEADING_NUMBER.match(cleaned_suffix)
            tail = cleaned_suffix[leading.end():].strip() if leading else ""
            tail = re.sub(r"^(?:[*_`~]\s*)+|(?:[*_`~]\s*)+$", "", tail).strip()
            if (
                not tail
                or tail.startswith(("%", "％"))
                or re.match(r"^[=：:]\s*[^0-9]*$", tail)
                or re.match(r"^[（(][^0-9]*[）)]$", tail)
            ):
                return numeric, status
    return None, "unparsed"


def _free_choice_token(value: Any) -> str:
    """Remove harmless presentation wrappers around one free-form choice."""

    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = re.sub(r"^(?:[*_`~]\s*)+", "", text)
    text = re.sub(r"(?:[*_`~]\s*)+$", "", text).strip()
    text = re.sub(
        r"^(?:图片|图示|图|圆圈对|diagram|option|选择|选项|选|编号|choice|pair|pairs)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    # A diagram answer is often written as ``图 **(2)**``: strip markup again
    # after removing the natural-language prefix, before removing parentheses.
    text = re.sub(r"^(?:[*_`~]\s*)+", "", text)
    text = re.sub(r"(?:[*_`~]\s*)+$", "", text).strip()
    text = re.sub(r"^[\(\[（【]\s*", "", text)
    text = re.sub(r"\s*[\)\]）】]$", "", text).strip()
    text = re.sub(r"^(?:[*_`~]\s*)+|(?:[*_`~]\s*)+$", "", text).strip()
    return text


_FREE_CHOICE_LEADING = re.compile(
    r"^\s*(?:[*_`~]\s*)*"
    r"(?:(?:图片|图示|图|圆圈对|diagram|option|选择|选项|选|编号|choice|pair|pairs)\s*)?"
    r"(?:[*_`~]\s*)*"
    r"[\(\[（【]?\s*"
    r"([+-]?\d+(?:\.\d+)?|[A-Za-z]+|是|否|正确|错误)"
    r"\s*[\)\]）】]?"
    r"(?:[*_`~]\s*)?",
    re.IGNORECASE,
)


def _free_leading_choice_value(
    body: str, choices: Sequence[str]
) -> Optional[str]:
    """Read one leading choice only when the remainder is non-ambiguous."""

    leading = _FREE_CHOICE_LEADING.match(body)
    if not leading:
        return None
    remainder = body[leading.end():]
    # ``A 和 B 都有道理`` / ``A or B`` is a comparison, not a selected
    # answer. A leading token is accepted only when the remainder does not
    # introduce an alternative or a second listed choice.
    has_alternative = bool(
        re.search(r"\b(?:or|and)\b|或|或者|还是|和", remainder, re.IGNORECASE)
    )
    has_other_choice = any(
        choice.casefold() != _free_choice_token(leading.group(1)).casefold()
        and re.search(
            rf"(?<![0-9A-Za-z]){re.escape(choice)}(?![0-9A-Za-z])",
            remainder,
            re.IGNORECASE,
        )
        for choice in choices
    )
    if has_alternative or has_other_choice:
        return None
    return _normalise_choice(_free_choice_token(leading.group(1)), choices)


def _free_extract_choice(body: str, choices: Sequence[str]) -> Optional[str]:
    """Find a listed choice token inside prose, including common aliases."""

    body = unicodedata.normalize("NFKC", str(body or ""))
    direct = _normalise_choice(_free_choice_token(body.strip()), choices)
    if direct is not None:
        return direct

    # Prefer an explicitly labelled answer, such as ``选择：是`` or
    # ``I choose B``. This prevents an option mentioned in an explanation from
    # being mistaken for the selected answer.
    marker = re.search(
        r"(?:answer|response|choice|selected|selection|choose|select|pick|"
        r"选择|选项|选|答案|回答|判断|选的是?|我选|我选择)"
        r"\s*(?:(?:is|为|是)\s*)?[:：=]?\s*"
        r"([^\r\n]*)",
        body,
        re.IGNORECASE,
    )
    if marker:
        marked_text = marker.group(1).strip()
        marked = _normalise_choice(_free_choice_token(marked_text), choices)
        if marked is not None:
            return marked
        marked = _free_leading_choice_value(marked_text, choices)
        if marked is not None:
            return marked

    # Accept a presentation wrapper such as ``图(2)``, ``**A**`` or
    # ``option B`` only at the start of an answer.  This is also why an
    # explanation mentioning another option is not searched globally.
    leading_value = _free_leading_choice_value(body, choices)
    if leading_value is not None:
        return leading_value

    # A label before the answer is common in free text (``同事：图(2)``).
    # Read only a suffix that contains one complete choice token, never an
    # arbitrary token buried in a long rationale.
    first_line = next((line.strip() for line in body.splitlines() if line.strip()), "")
    separator = re.search(r"[:：=]", first_line)
    if separator:
        suffix = first_line[separator.end():].strip()
        value = _free_leading_choice_value(suffix, choices)
        if value is not None:
            return value

    return None


def _free_extract_ios_pair(body: str, choices: Sequence[str]) -> Optional[str]:
    """Extract one IOS whole-diagram label without accepting circle pairs."""
    text = unicodedata.normalize("NFKC", str(body or ""))
    text = re.sub(r"[*_`]", "", text)
    label = (r"(?:\b(?:pair|figure|diagram|picture|image|option|choose|select|pick|choice|answer)\b"
             r"|图(?:片|示|形)?|对应|选择|选项|我选|选|答案)")
    number = r"\(?\s*([1-7])(?![\d.])\s*\)?"
    marked = re.compile(label + r"\s*(?:(?:is|number|no\.?|为|是|编号)\s*)?[:：=#]?\s*(?:第\s*)?" + number, re.I)
    leading = re.compile(r"^\s*" + number)
    alternatives = re.compile(
        r"^\s*\(?\s*(?:or\b|and\b|或(?:者)?|和|至|到|[/,，、&—–~-])\s*"
        r"(?:" + label + r"\s*)*\(?\s*[+-]?\d", re.I)
    for line in text.splitlines():
        line = line.strip()
        if not line or re.match(r"^[（(]?\s*(?:注[:：]|note\b|理由[:：]|解释[:：]|reason(?:ing)?[:：]|explanation[:：]|rationale[:：])", line, re.I):
            continue
        # A stated selection outranks numbers appearing in the explanation.
        # Words such as 'and'/'或' are alternatives only next to another code.
        match = marked.search(line)
        target = line
        if match is None:
            target = re.split(r"[:：→]|[—–]|\s+-\s+", line, maxsplit=1)[-1].strip()
            match = leading.match(target)
        if match is None:
            continue
        tail = target[match.end():]
        other_item = re.match(
            r"^[,，;；]\s*(?:(?:\d+\s*[.、:：])|"
            r"(?:(?:colleagues?|family(?:\s+members?)?|relatives?|friends?|同事|家庭成员|家人|亲戚|朋友)\s*[:：—–-]))",
            tail, re.I)
        if alternatives.match(tail) or other_item:
            return None
        return _normalise_choice(match.group(1), choices)
    return None


def _free_split_markdown_row(line: str) -> Optional[List[str]]:
    """Split a Markdown table row while ignoring separator rows."""

    stripped = str(line or "").strip()
    if "|" not in stripped:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    if len(cells) < 2 or all(_FREE_TABLE_DIVIDER.fullmatch(cell) for cell in cells):
        return None
    return cells


def _free_table_rows(text: str) -> Dict[int, List[tuple[List[str], List[str]]]]:
    """Collect numbered Markdown rows and the nearest table header."""

    rows: Dict[int, List[tuple[List[str], List[str]]]] = {}
    header: Optional[List[str]] = None
    in_table = False
    for raw_line in str(text or "").splitlines():
        cells = _free_split_markdown_row(raw_line)
        if cells is None:
            if not raw_line.strip():
                header = None
                in_table = False
            continue
        number = _free_table_item_number(cells[0])
        if number is not None:
            if number >= 1:
                rows.setdefault(number, []).append((cells, list(header or [])))
                in_table = True
            continue
        if in_table:
            # A non-numbered row after data is not a new header.
            continue
        header = cells
        in_table = True
    return rows


def _free_header_is_answer(value: str) -> bool:
    label = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return any(token in label for token in _FREE_ANSWER_HEADER_HINTS)


def _free_header_is_question(value: str) -> bool:
    label = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return any(token in label for token in _FREE_QUESTION_HEADER_HINTS)


def _free_numeric_answer(
    slot: ScaleSlot,
    answer: str,
    numeric: float,
    status: str,
) -> ParsedAnswer:
    error = None
    if slot.minimum is not None and numeric < float(slot.minimum):
        status, error = "out_of_range", f"Score {numeric:g} is below the declared minimum."
    elif slot.maximum is not None and numeric > float(slot.maximum):
        status, error = "out_of_range", f"Score {numeric:g} is above the declared maximum."
    return ParsedAnswer(
        answer=answer,
        score=numeric,
        slot_id=slot.slot_id,
        parse_status=status,
        parse_error=error,
    )


def _free_extract_table_answer(
    slot: ScaleSlot,
    rows: Sequence[tuple[List[str], List[str]]],
) -> Optional[ParsedAnswer]:
    """Decode a numbered Markdown row without reading later summary text."""

    for cells, headers in rows:
        if len(cells) < 2:
            continue
        data_indices = list(range(1, len(cells)))
        preferred = [
            index for index in data_indices
            if index < len(headers) and _free_header_is_answer(headers[index])
        ]
        fallback = [
            index for index in data_indices
            if not (index < len(headers) and _free_header_is_question(headers[index]))
        ]
        indices = preferred + [index for index in fallback if index not in preferred]

        if _is_open_text_slot(slot):
            for index in indices:
                cell = str(cells[index] or "").strip()
                if cell:
                    return ParsedAnswer(
                        answer=cell,
                        score=None,
                        slot_id=slot.slot_id,
                        parse_status="text_response",
                    )
            continue

        if slot.response_type == "ios_pair" and slot.choices:
            for index in indices:
                chosen = _free_extract_ios_pair(cells[index], slot.choices)
                if chosen is None:
                    continue
                return ParsedAnswer(
                    answer=chosen,
                    score=_numeric_score(chosen),
                    slot_id=slot.slot_id,
                    parse_status="parsed",
                )
            continue

        if slot.response_type == "pair_code" and slot.choices:
            for index in indices:
                chosen = _free_extract_pair_code(cells[index], slot.choices)
                if chosen is None:
                    continue
                return ParsedAnswer(
                    answer=chosen,
                    score=None,
                    slot_id=slot.slot_id,
                    parse_status="parsed",
                )
            continue

        if slot.response_type == "choice" and slot.choices:
            for index in indices:
                chosen = _free_extract_choice(cells[index], slot.choices)
                if chosen is None:
                    continue
                return ParsedAnswer(
                    answer=chosen,
                    score=_numeric_score(chosen),
                    slot_id=slot.slot_id,
                    parse_status="parsed",
                )
            continue

        unresolved: Optional[ParsedAnswer] = None
        text_fallback: Optional[str] = None
        for index in indices:
            cell = str(cells[index] or "").strip()
            interval = _free_extract_interval(cell)
            if interval is not None and slot.response_type != "text":
                return _free_range_answer(slot, cell, interval)
            numeric, status = _free_extract_numeric(cell)
            if numeric is not None:
                return _free_numeric_answer(slot, cell, numeric, status)
            if slot.response_type == "text" and cell and text_fallback is None:
                text_fallback = cell
            if status == "range_unresolved" and unresolved is None:
                unresolved = ParsedAnswer(
                    answer=cell,
                    score=None,
                    slot_id=slot.slot_id,
                    parse_status="range_unresolved",
                    parse_error="The response gives multiple possible scores; no single score was guessed.",
                )
        if unresolved is not None:
            if slot.response_type == "text":
                return _free_text_response(slot, str(unresolved.answer))
            return unresolved
        if text_fallback is not None:
            return _free_text_response(slot, text_fallback)
    return None


def _free_parse_value(slot: ScaleSlot, value: Any) -> ParsedAnswer:
    """Decode one self-formatted free-mode value without trusting its schema.

    This is used only for the narrow JSON/table recovery path. The free
    contract did not ask the model for JSON, so a string value must go through
    the same conservative rules as an ordinary numbered answer.
    """

    if value is None:
        return ParsedAnswer(slot_id=slot.slot_id, parse_status="missing")
    if _is_open_text_slot(slot):
        if isinstance(value, str) and value.strip():
            return ParsedAnswer(
                answer=value,
                score=None,
                slot_id=slot.slot_id,
                parse_status="text_response",
            )
        return ParsedAnswer(
            answer=value,
            slot_id=slot.slot_id,
            parse_status="unparsed",
            parse_error="The open-text answer is empty or not text.",
        )
    if slot.response_type == "ios_pair" and slot.choices:
        chosen = _free_extract_ios_pair(str(value), slot.choices)
        if chosen is not None:
            return ParsedAnswer(
                answer=chosen,
                score=_numeric_score(chosen),
                slot_id=slot.slot_id,
                parse_status="parsed",
            )
        return ParsedAnswer(
            answer=value,
            slot_id=slot.slot_id,
            parse_status="unparsed",
            parse_error="No valid IOS pair-diagram label appears in this answer.",
        )
    if slot.response_type == "choice" and slot.choices:
        if all(_numeric_score(choice) is not None for choice in slot.choices):
            interval = _free_extract_interval(str(value))
            if interval is not None:
                return _free_range_answer(slot, value, interval)
        chosen = _free_extract_choice(str(value), slot.choices)
        if chosen is not None:
            return ParsedAnswer(
                answer=chosen,
                score=_numeric_score(chosen),
                slot_id=slot.slot_id,
                parse_status="parsed",
            )
        return ParsedAnswer(
            answer=value,
            slot_id=slot.slot_id,
            parse_status="unparsed",
            parse_error="No listed choice appears in this answer.",
        )
    if slot.response_type == "pair_code" and slot.choices:
        chosen = _free_extract_pair_code(str(value), slot.choices)
        if chosen is not None:
            return ParsedAnswer(
                answer=chosen,
                score=None,
                slot_id=slot.slot_id,
                parse_status="parsed",
            )
        return ParsedAnswer(
            answer=value,
            slot_id=slot.slot_id,
            parse_status="unparsed",
            parse_error="No valid circle-pair code appears in this answer.",
        )

    numeric, status = _free_extract_numeric(str(value))
    interval = _free_extract_interval(str(value))
    if interval is not None and slot.response_type != "text":
        return _free_range_answer(slot, value, interval)
    if status == "range_unresolved":
        if slot.response_type == "text":
            return _free_text_response(slot, str(value))
        return ParsedAnswer(
            answer=value,
            slot_id=slot.slot_id,
            parse_status=status,
            parse_error="The response gives multiple possible scores; no single score was guessed.",
        )
    if numeric is None:
        if slot.response_type == "text" and str(value).strip():
            return _free_text_response(slot, str(value))
        return ParsedAnswer(
            answer=value,
            slot_id=slot.slot_id,
            parse_status="unparsed",
            parse_error="No numeric score appears in this answer.",
        )
    return _free_numeric_answer(slot, value, numeric, status)


def _free_decode_body(lines: Sequence[str]) -> str:
    """Use the first answer paragraph for decoding, excluding later summaries."""

    kept: List[str] = []
    started = False
    for raw_line in lines:
        line = str(raw_line or "").rstrip()
        if not line.strip():
            if started:
                break
            continue
        kept.append(line)
        started = True
    return "\n".join(kept).strip()


_FREE_POSITIONAL_TOKEN = re.compile(r"[^\s,，;；、|]+")
_FREE_POSITIONAL_SEPARATORS = re.compile(r"^[\s,，;；、|]+$")


def _free_positional_tokens(text: str) -> Optional[List[str]]:
    """Split a bare answer sequence without accepting numbered-item syntax.

    This is intentionally stricter than a generic number search.  It accepts
    only tokens separated by whitespace or simple list separators, so prose,
    item labels (``1. 5``), ranges (``3-4``), and explanations cannot be
    silently converted into positional answers.
    """

    normalized = unicodedata.normalize("NFKC", str(text or "")).strip()
    normalized = normalized.replace("−", "-").replace("＋", "+")
    fenced = re.fullmatch(
        r"```(?:text|txt|plain)?\s*([\s\S]*?)\s*```",
        normalized,
        re.IGNORECASE,
    )
    if fenced:
        normalized = fenced.group(1).strip()
    if not normalized:
        return None

    # A line beginning with an item label belongs to the ordinary labelled
    # decoder.  Refusing it here prevents ``1. 5\n2. 3`` from being mistaken
    # for two positional values when the count happens to match.
    if re.search(r"(?m)^\s*\d+\s*[.、．)）:]\s+", normalized):
        return None

    matches = list(_FREE_POSITIONAL_TOKEN.finditer(normalized))
    if not matches:
        return None
    cursor = 0
    tokens: List[str] = []
    for match in matches:
        between = normalized[cursor:match.start()]
        if between and not _FREE_POSITIONAL_SEPARATORS.fullmatch(between):
            return None
        tokens.append(match.group(0).strip())
        cursor = match.end()
    tail = normalized[cursor:]
    if tail and not _FREE_POSITIONAL_SEPARATORS.fullmatch(tail):
        return None
    return tokens


def _free_positional_answers(
    text: str, slots: Sequence[ScaleSlot]
) -> Optional[List[ParsedAnswer]]:
    """Validate a complete unlabeled answer sequence against every slot.

    Positional recovery is safe only when the whole response can be validated
    at once.  A single mismatch rejects the entire recovery; the caller then
    keeps the raw response and leaves item scores unresolved.
    """

    tokens = _free_positional_tokens(text)
    if tokens is None or len(tokens) != len(slots):
        return None
    if any(_is_open_text_slot(slot) for slot in slots):
        return None

    answers: List[ParsedAnswer] = []
    for slot, token in zip(slots, tokens):
        if slot.response_type == "ios_pair" and slot.choices:
            chosen = _free_extract_ios_pair(token, slot.choices)
            if chosen is None:
                return None
            answers.append(
                ParsedAnswer(
                    answer=chosen,
                    score=_numeric_score(chosen),
                    slot_id=slot.slot_id,
                    parse_status="parsed",
                )
            )
            continue

        if slot.response_type == "pair_code" and slot.choices:
            chosen = _free_extract_pair_code(token, slot.choices)
            if chosen is None:
                return None
            answers.append(
                ParsedAnswer(
                    answer=chosen,
                    score=None,
                    slot_id=slot.slot_id,
                    parse_status="parsed",
                )
            )
            continue

        if slot.response_type == "choice" and slot.choices:
            chosen = _free_extract_choice(token, slot.choices)
            if chosen is None:
                return None
            answers.append(
                ParsedAnswer(
                    answer=chosen,
                    score=_numeric_score(chosen),
                    slot_id=slot.slot_id,
                    parse_status="parsed",
                )
            )
            continue

        numeric, status = _free_extract_numeric(token)
        if numeric is None or status != "parsed":
            return None
        answers.append(_free_numeric_answer(slot, token, numeric, status))
    return answers


def _parse_free_answers_legacy(
    text: str,
    slots: Sequence[ScaleSlot],
    *,
    allow_positional_fallback: bool = False,
) -> ParseResult:
    """Read scores opportunistically while preserving non-numeric answer text.

    This parser runs after the model responds; it adds no answer requirements
    to the prompt. The complete model response is also retained in the trial.
    """

    n_items = len(slots)
    answers: List[ParsedAnswer] = [
        ParsedAnswer(slot_id=slot.slot_id, parse_status="missing") for slot in slots
    ]
    if n_items == 0:
        return ParseResult(answers=answers, status="ok")

    # Locate explicitly labeled answers when the model happens to return them.
    # Numbering is never required in the request; unlabeled text is considered
    # only when exact paragraph boundaries make a one-to-one mapping possible.
    body_by_item: Dict[int, List[str]] = {}
    table_rows = _free_table_rows(text)
    current: Optional[int] = None
    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        table_cells = _free_split_markdown_row(line)
        if table_cells is not None:
            table_number = _free_table_item_number(table_cells[0])
            if table_number is not None and 1 <= table_number <= n_items:
                # Keep a fallback body for malformed tables, but prefer the
                # structured row decoder below when a score column exists.
                current = table_number
                body_by_item.setdefault(current, []).append(" | ".join(table_cells[1:]))
                continue
        match = _FREE_ITEM_LINE.match(line) or _FREE_ITEM_LABEL.match(line)
        if match:
            number = int(match.group(1))
            if 1 <= number <= n_items:
                current = number
                body_by_item.setdefault(number, []).append(match.group(2))
                continue
        if current is not None:
            body_by_item.setdefault(current, []).append(line)

    if (
        not body_by_item
        and not table_rows
        and all(slot.response_type == "text" for slot in slots)
    ):
        raw = str(text or "").strip()
        text_blocks = _unlabeled_text_blocks(raw, n_items)
        if n_items == 1 and raw:
            text_blocks = [raw]
        if text_blocks is not None:
            for index, (slot, body) in enumerate(zip(slots, text_blocks)):
                if _is_open_text_slot(slot):
                    answers[index] = _free_text_response(slot, body)
                    continue
                numeric, status = _free_extract_numeric(body)
                if numeric is not None:
                    answers[index] = _free_numeric_answer(slot, body, numeric, status)
                else:
                    answers[index] = _free_text_response(slot, body)
            return ParseResult(
                answers=answers,
                status="ok",
                recovery="free-unlabeled-text-v1",
            )

    for index, slot in enumerate(slots, start=1):
        table_answer = _free_extract_table_answer(slot, table_rows.get(index, []))
        if table_answer is not None:
            answers[index - 1] = table_answer
            continue

        body_lines = body_by_item.get(index, [])
        body = "\n".join(body_lines).strip()
        if not body:
            answers[index - 1] = ParsedAnswer(
                slot_id=slot.slot_id,
                parse_status="missing",
                parse_error="No answer could be located for this item in display order.",
            )
            continue

        # Open-ended tasks (for example Kohlberg's moral-judgement stories)
        # are not rating items. Preserve the complete response as text and
        # never mine numbers from its rationale for a fictitious score.
        if _is_open_text_slot(slot):
            answers[index - 1] = ParsedAnswer(
                answer=body,
                score=None,
                slot_id=slot.slot_id,
                parse_status="text_response",
            )
            continue

        if slot.response_type == "ios_pair" and slot.choices:
            chosen = _free_extract_ios_pair(body, slot.choices)
            if chosen is None:
                answers[index - 1] = ParsedAnswer(
                    answer=body,
                    slot_id=slot.slot_id,
                    parse_status="unparsed",
                    parse_error="No valid IOS pair-diagram label appears in this answer.",
                )
            else:
                answers[index - 1] = ParsedAnswer(
                    answer=chosen,
                    score=_numeric_score(chosen),
                    slot_id=slot.slot_id,
                    parse_status="parsed",
                )
            continue

        if slot.response_type == "choice" and slot.choices:
            chosen = _free_extract_choice(body, slot.choices)
            if chosen is None:
                answers[index - 1] = ParsedAnswer(
                    answer=body,
                    slot_id=slot.slot_id,
                    parse_status="unparsed",
                    parse_error="No listed choice appears in this answer.",
                )
            else:
                answers[index - 1] = ParsedAnswer(
                    answer=chosen,
                    score=_numeric_score(chosen),
                    slot_id=slot.slot_id,
                    parse_status="parsed",
                )
            continue

        if slot.response_type == "pair_code" and slot.choices:
            chosen = _free_extract_pair_code(body, slot.choices)
            if chosen is None:
                answers[index - 1] = ParsedAnswer(
                    answer=body,
                    slot_id=slot.slot_id,
                    parse_status="invalid_pair_code",
                    parse_error="No valid circle-pair code appears in this answer.",
                )
            else:
                answers[index - 1] = ParsedAnswer(
                    answer=chosen,
                    score=None,
                    slot_id=slot.slot_id,
                    parse_status="parsed",
                )
            continue

        # Decode the first answer paragraph. The complete body is still kept
        # in ``answer`` when no table-specific answer cell is available, but
        # trailing totals/averages must not become the final item's score.
        decode_body = _free_decode_body(body_lines) or body
        interval = _free_extract_interval(decode_body)
        if interval is not None and slot.response_type != "text":
            answers[index - 1] = _free_range_answer(slot, body, interval)
            continue
        numeric, status = _free_extract_numeric(decode_body)
        if status == "range_unresolved" and slot.response_type == "text":
            answers[index - 1] = _free_text_response(slot, body)
            continue
        if status == "range_unresolved":
            answers[index - 1] = ParsedAnswer(
                answer=body,
                score=None,
                slot_id=slot.slot_id,
                parse_status=status,
                parse_error="The response gives multiple possible scores; no single score was guessed.",
            )
            continue
        if numeric is None:
            if slot.response_type == "text":
                answers[index - 1] = _free_text_response(slot, body)
                continue
            answers[index - 1] = ParsedAnswer(
                answer=body,
                slot_id=slot.slot_id,
                parse_status="unparsed",
                parse_error="No numeric score appears in this answer.",
            )
            continue

        answers[index - 1] = _free_numeric_answer(slot, body, numeric, status)

    def usable(index: int) -> bool:
        """A slot counts as read when it holds any concrete answer.

        Choice slots carry a letter rather than a numeric score, so keying
        completion off ``score`` alone would report a fully correct response as
        partial.
        """

        answer = answers[index]
        return answer.parse_status in FREE_STATUS_READABLE

    read = sum(1 for index in range(n_items) if usable(index))
    recovery = FREE_PARSER_VERSION
    # Positional recovery is post-response parsing only. It accepts a sequence
    # only when every item can be mapped conservatively; no such format is
    # requested from the model.
    if (
        allow_positional_fallback
        and read == 0
        and not body_by_item
        and not table_rows
    ):
        positional = _free_positional_answers(text, slots)
        if positional is not None:
            answers = positional
            recovery = f"{FREE_PARSER_VERSION}-positional-v1"
            read = sum(
                1 for answer in answers if answer.parse_status in FREE_STATUS_READABLE
            )

    # A model that was never asked for JSON may still return it — bare or in a
    # fenced block. Merge only slots that the free-text decoder could not read;
    # a valid free-text answer must never be replaced by a guessed JSON value.
    fallback = parse_answers(str(text or ""), n_items)
    if fallback.status != "invalid_json":
        merged = False
        for index, fallback_answer in enumerate(fallback.answers):
            if index >= len(slots) or usable(index):
                continue
            fallback_answer = _free_parse_value(
                slots[index], fallback_answer.answer
            )
            fallback_usable = fallback_answer.parse_status in FREE_STATUS_READABLE
            # Preserve supplied but undecodable wording as well as readable
            # values; it remains available for later manual coding.
            fallback_has_answer = (
                fallback_answer.answer is not None
                and fallback_answer.parse_status != "missing"
            )
            if fallback_usable or fallback_has_answer:
                answers[index] = fallback_answer
                merged = True
        if merged:
            recovery = "free_text_json_fallback"

    read = sum(1 for i in range(n_items) if usable(i))
    if read == n_items:
        result_status = "ok"
    elif read:
        result_status = "partial"
    else:
        result_status = "unparsed"
    error = None
    if result_status == "partial":
        missing = [i + 1 for i in range(n_items) if not usable(i)]
        error = "No answer could be read for display_index: " + ", ".join(map(str, missing)) + "."
    elif result_status == "unparsed":
        error = "No answer could be read for any item in this response."
    return ParseResult(answers=answers, status=result_status, error=error, recovery=recovery)


def parse_free_answers(text: str, slots: Sequence[ScaleSlot], *, allow_positional_fallback: bool = False) -> ParseResult:
    from .response_decoder import decode_free_response
    return decode_free_response(text, slots, allow_positional_fallback=allow_positional_fallback)


def parse_kohlberg_free_answers(text: str, slots: Sequence[ScaleSlot]) -> ParseResult:
    """Safely segment free-form Kohlberg story responses.

    A Kohlberg story can contain internal prompts numbered 1, 1a, 2, and so
    on. The request preserves the source text without adding story labels.
    This decoder uses explicit boundaries if the model supplies them; otherwise
    it maps blank-line-separated prose only when the block count exactly
    matches the number of open-ended items. The full raw response is retained.
    """

    from .response_decoder import _plain, _OUTER, _NUMBERS, decode_interview_rows
    source_rows = decode_interview_rows(text, slots)
    if source_rows is not None:
        return source_rows
    n_items = len(slots)
    answers: List[ParsedAnswer] = [
        ParsedAnswer(slot_id=slot.slot_id, parse_status="missing") for slot in slots
    ]
    if n_items == 0:
        return ParseResult(answers=answers, status="ok", recovery="kohlberg-story-v1")

    chinese_numbers = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    body_by_story: Dict[int, List[str]] = {}
    current: Optional[int] = None
    marker_count: Dict[int, int] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        clean_line = _plain(line)
        marker = _KOHLBERG_STORY_MARKER.match(clean_line) or _OUTER.match(clean_line)
        if marker:
            raw_index = marker.group(1)
            index = (
                int(raw_index)
                if raw_index.isdigit()
                else chinese_numbers.get(raw_index, _NUMBERS.get(raw_index.casefold()))
            )
            if index is not None and 1 <= index <= n_items:
                current = index
                marker_count[index] = marker_count.get(index, 0) + 1
                body_by_story.setdefault(index, [])
                remainder = marker.group(2).strip() if marker.re.groups >= 2 else ""
                if remainder:
                    body_by_story[index].append(remainder)
                continue
        if current is not None:
            body_by_story.setdefault(current, []).append(line)

    if not body_by_story:
        raw = str(text or "").strip()
        text_blocks = _unlabeled_text_blocks(raw, n_items)
        if n_items == 1 and raw:
            text_blocks = [raw]
        if text_blocks is not None and all(slot.response_type == "text" for slot in slots):
            for index, (slot, body) in enumerate(zip(slots, text_blocks)):
                answers[index] = _free_text_response(slot, body)
            return ParseResult(
                answers=answers,
                status="ok",
                recovery="kohlberg-unlabeled-text-v1",
            )
        error = "No unambiguous story segmentation was found; raw_response was retained."
        for answer in answers:
            answer.parse_error = error
        return ParseResult(
            answers=answers,
            status="unparsed",
            error=error,
            recovery="kohlberg-story-v1",
        )

    # Workbook rows may be continuation sections within one named story.
    # Incomplete/repeated story labels alone cannot identify those rows.
    if set(body_by_story) != set(range(1, n_items + 1)) or any(count > 1 for count in marker_count.values()):
        return ParseResult(answers=answers, status="unparsed",
                           error="Story labels do not establish a unique boundary for every source row; raw_response was retained.",
                           recovery="kohlberg-story-v1")

    for index, slot in enumerate(slots, start=1):
        body = "\n".join(body_by_story.get(index, [])).strip()
        if body:
            answers[index - 1] = ParsedAnswer(
                answer=body,
                slot_id=slot.slot_id,
                parse_status="text_response",
            )
        else:
            answers[index - 1] = ParsedAnswer(
                slot_id=slot.slot_id,
                parse_status="missing",
                parse_error="No text was found after the explicit story boundary.",
            )

    readable = sum(answer.parse_status == "text_response" for answer in answers)
    if readable == n_items:
        status = "ok"
        error = None
    elif readable:
        status = "partial"
        missing = [str(index) for index, answer in enumerate(answers, start=1) if answer.parse_status != "text_response"]
        error = "No story-level answer could be read for: " + ", ".join(missing) + "."
    else:
        status = "unparsed"
        error = "No story-level answer could be read from the response."
    return ParseResult(
        answers=answers,
        status=status,
        error=error,
        recovery="kohlberg-story-v1",
    )


def parse_scores(text: str, n_items: int) -> List[Optional[float]]:
    """Backward-compatible numeric view of :func:`parse_answers`."""

    return [answer.score for answer in parse_answers(text, n_items).answers]


def shuffle_order(n: int, rng: Optional[random.Random] = None) -> List[int]:
    order = list(range(n))
    (rng or random).shuffle(order)
    return order


def remap_to_original(shuffled_values: Sequence[Any], order: Sequence[int]) -> List[Any]:
    """Map display-order values back to their original worksheet positions."""

    result: List[Any] = [None] * len(order)
    for display_pos, orig_i in enumerate(order):
        if display_pos < len(shuffled_values):
            result[orig_i] = shuffled_values[display_pos]
    return result


def build_result_payload(
    *,
    scale_name: str,
    language: str,
    provider: str,
    model: str,
    temperature: float,
    order_id: int,
    seed: Optional[int],
    order: Sequence[int],
    shuffle_enabled: bool,
    sheet: ScaleSheet,
    answers_original: Sequence[ParsedAnswer],
    parse_result: ParseResult,
    raw_response: str,
    prompt: str,
    display_plan: Optional[DisplayPlan] = None,
    model_metadata: Optional[Dict[str, Any]] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    request_snapshot: Optional[Dict[str, Any]] = None,
    response_snapshot: Any = None,
    run_id: Optional[str] = None,
    condition_mode: str = "custom",
    order_strategy: str = "per_scale",
    pair_id: Optional[int] = None,
    condition_descriptor: Optional[Dict[str, Any]] = None,
    condition_hash: Optional[str] = None,
    execution_hash: Optional[str] = None,
    prompt_contract: str = DEFAULT_PROMPT_CONTRACT,
    cultural_identity: str = "none",
    response_status: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a lossless result record while preserving legacy numeric ``scores``."""

    def has_answer(value: Any) -> bool:
        return value is not None and not (isinstance(value, str) and not value.strip())

    items: List[Dict[str, Any]] = []
    canonical_slots = sheet.slots
    block_id_by_slot = {
        slot.slot_id: block.block_id
        for block in sheet.blocks
        for slot in block.slots
    }
    for i, question in enumerate(sheet.questions):
        answer = (
            answers_original[i]
            if i < len(answers_original) and isinstance(answers_original[i], ParsedAnswer)
            else ParsedAnswer(parse_status="missing")
        )
        item: Dict[str, Any] = {
            "index": i + 1,
            "question": question,
            "answer": answer.answer,
            "score": answer.score,
            "parse_status": answer.parse_status,
            "parse_error": answer.parse_error,
            "mapping_method": answer.mapping_method,
            "range_text": answer.range_text,
            "range_lower": answer.range_lower,
            "range_upper": answer.range_upper,
            "range_unit": answer.range_unit,
            "answer_present": has_answer(answer.answer),
            "score_present": answer.score is not None,
        }
        if sheet.is_profiled and i < len(canonical_slots):
            slot = canonical_slots[i]
            item.update(slot.metadata_dict())
            # Stable IDs are canonical profile IDs, not model-provided text.
            item["slot_id"] = slot.slot_id
            item["block_id"] = block_id_by_slot.get(slot.slot_id)
        items.append(item)

    identity_metadata = dict(model_metadata or {
        "provider": provider,
        "requested_model": model,
    })
    model_profile = identity_metadata.pop("model_profile", None)
    # The complete profile is stored once at the result top level. Keep the
    # audit trail in model_identity but avoid embedding another full copy.
    parameter_audit = identity_metadata.get("parameter_audit")
    if isinstance(parameter_audit, dict) and "profile" in parameter_audit:
        parameter_audit = dict(parameter_audit)
        parameter_audit.pop("profile", None)
        identity_metadata["parameter_audit"] = parameter_audit

    payload: Dict[str, Any] = {
        "result_schema_version": "scale-result-v3" if sheet.is_profiled else "scale-result-v2",
        "response_parser": {
            "version": (
                FREE_PARSER_VERSION
                if is_free_prompt_contract(prompt_contract)
                else (PROFILE_PARSER_VERSION if sheet.is_profiled else PARSER_VERSION)
            ),
            "status": parse_result.status,
            "error": parse_result.error,
            "recovery": parse_result.recovery,
        },
        "response_status": response_status or (
            parse_result.status
            if parse_result.status in {"ok", "partial"}
            else "unparsed"
        ),
        "item_count": len(items),
        # These counts intentionally answer different questions. A retained
        # raw answer may be undecodable, and open text or an interval is
        # readable without supplying a single numeric score.
        "answered_item_count": sum(1 for item in items if item["answer_present"]),
        "read_item_count": sum(
            1
            for item in items
            if item.get("parse_status") in FREE_STATUS_READABLE
        ),
        "scored_item_count": sum(
            1 for item in items if item.get("score_present")
        ),
        "all_items_read": (
            len(items) == 0
            or sum(
                1
                for item in items
                if item.get("parse_status")
                in FREE_STATUS_READABLE
            )
            == len(items)
        ),
        "scale_name": scale_name,
        "scale_title": sheet.title,
        "language": language,
        "provider": provider,
        "model": model,
        "condition_mode": condition_mode,
        "condition_descriptor": dict(condition_descriptor or {}),
        "condition_config_sha256": (condition_descriptor or {}).get(
            "requested_config_sha256"
        ),
        "trial_key": f"{scale_name}/{language}/{int(order_id):04d}",
        "order_strategy": order_strategy,
        "pair_id": pair_id,
        "paired_trial_key": (
            f"{scale_name}/{int(pair_id):04d}" if pair_id is not None else None
        ),
        # Requested model and provider are retained for backwards-compatible
        # readers; this nested record captures the identity returned by the API
        # when the provider exposes it.
        "model_identity": identity_metadata,
        "attempt_id": identity_metadata.get("attempt_id"),
        "client_request_id": identity_metadata.get("client_request_id"),
        "provider_request_id": identity_metadata.get("provider_request_id"),
        "network_snapshot_id": identity_metadata.get("network_snapshot_id"),
        "model_catalog_snapshot_id": identity_metadata.get(
            "model_catalog_snapshot_id"
        ),
        "account_snapshot_id": identity_metadata.get("account_snapshot_id"),
        "temperature": temperature,
        "order_id": order_id,
        "shuffle_enabled": shuffle_enabled,
        "shuffle_seed": seed,
        "shuffle_order": [i + 1 for i in order],
        "instruction": sheet.instruction,
        "images": [image.metadata() for image in sheet.images],
        "items": items,
        "scores": [item["score"] for item in items],
        "raw_response": raw_response,
        "prompt": prompt,
        "prompt_sha256": canonical_hash(prompt),
        # Record which prompt condition was used. Answer text is preserved
        # independently of any optional score parsing.
        "prompt_contract": normalize_prompt_contract(prompt_contract),
        "cultural_identity": normalize_cultural_identity(cultural_identity),
        "cultural_identity_prompt": cultural_identity_prompt(cultural_identity, language),
        "parser_version": (
            FREE_PARSER_VERSION
            if is_free_prompt_contract(prompt_contract)
            else (PROFILE_PARSER_VERSION if sheet.is_profiled else PARSER_VERSION)
        ),
    }

    # These fields are optional so old result fixtures and callers remain
    # readable. Formal runs should always provide them to distinguish the
    # configured request from what the provider actually returned.
    if generation_config is not None:
        payload["generation_config"] = dict(generation_config)
    if model_profile is not None:
        # Keep the complete resolved/documented profile at the top level so
        # analysis code need not know the nesting of model_identity.
        payload["model_profile"] = model_profile
    if request_snapshot is not None:
        payload["request_snapshot"] = dict(request_snapshot)
    if response_snapshot is not None:
        payload["response_snapshot"] = response_snapshot
    if request_snapshot is not None:
        payload["request_sha256"] = canonical_hash(request_snapshot)
    if response_snapshot is not None:
        payload["response_sha256"] = canonical_hash(response_snapshot)
    if run_id:
        payload["run_id"] = run_id
    if condition_hash:
        payload["condition_hash"] = condition_hash
    if execution_hash:
        payload["execution_hash"] = execution_hash

    if sheet.is_profiled:
        canonical_index = {slot.slot_id: index for index, slot in enumerate(canonical_slots)}
        blocks: List[Dict[str, Any]] = []
        for block in sheet.blocks:
            block_slots = []
            for slot in block.slots:
                index = canonical_index[slot.slot_id]
                block_slots.append(dict(items[index]))
            blocks.append(
                {
                    "block_id": block.block_id,
                    "label": block.label,
                    "context": block.context,
                    "instruction": block.instruction,
                    "shuffle_policy": block.shuffle_policy,
                    "metadata": dict(block.metadata),
                    "slots": block_slots,
                }
            )
        payload["scale_profile"] = {
            "id": sheet.profile_id,
            "version": sheet.profile_version,
            "label": sheet.profile_label,
            "validation_status": "ok" if not sheet.profile_error else "invalid_layout",
            "validation_error": sheet.profile_error,
            "expected_slots": sheet.n_items,
            "shuffle_supported": sheet.shuffle_supported,
            "default_shuffle": sheet.default_shuffle,
        }
        payload["blocks"] = blocks
        plan = display_plan or build_display_plan(sheet, False)
        payload["display_plan"] = plan.metadata()
    return payload
