"""Decode observed free responses using source questions and local task boundaries.

This module operates on returned text only. It never changes model prompts.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .scale_loader import ScaleSlot


def _plain(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).replace("−", "-")
    value = value.replace("’", "'").replace("‘", "'")
    value = re.sub(r"[*_`]", "", value)
    return re.sub(r"^\s*(?:#{1,6}\s*|[-+•]\s+)", "", value).strip()


def _signature(value: str) -> str:
    value = re.sub(r"[（(][^()（）]*[）)]", "", _plain(value))
    value = re.sub(r"^\d+\s*[.、):：]\s*", "", value)
    return "".join(c for c in value.casefold() if c.isalnum())


def _source_line(value: str) -> str:
    """Remove presentation prefixes without normalising the answer's text."""
    value = re.sub(r"^\s*(?:#{1,6}\s*|[-+•]\s+)", "", value)
    return re.sub(r"^[*_`\s]+", "", value)


_NUMBERS = dict(zip(("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"), range(1, 11)))
_NUMBERS.update(dict(zip("一二三四五六七八九", range(1, 10))))
_NUMBERS["十"] = 10
_OUTER = re.compile(r"^(?:情境|场景|故事|episode|situation|scenario|story|dilemma)\s*(?:第|#)?\s*(\d+|[一二三四五六七八九十]+|one|two|three|four|five|six|seven|eight|nine|ten)(?=\s|[:：.、]|$)", re.I)
_ITEM = re.compile(r"^(?:(?:item|question|q)\s*|第\s*)?[（(]?(\d+)\s*(?:题|问)?(?:\.(?!\d)|[、)）:：])\s*(.*)$", re.I)
_SECTION = re.compile(r"^(III|II|I|A|B|C)\s*[.:：、]\s*(.*)$", re.I)
_SCALAR = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*[%％分人]|\s*/\s*\d+)?(?:\s*(?:[=:(（—–,;。；-]|[.!?]\s*$|$))")
_SUMMARY = re.compile(r"summary|summari[sz]|总结|汇总|整理|合成.*答案|最终(?:答案|分数|回答)|final\s+(?:answers?|scores?|list)|百分比列表", re.I)
_GROUPNESS_HEADING = re.compile(r"^(?:group(?:ness|\s+perception)|群体(?:实体性|感知|知觉|性)?)(?:判断|评分)?\s*(?:[:：]\s*(.*)|$)", re.I)

# These identify a value the respondent explicitly selected, including a
# final scalar supplied after discussing an interval. They do not infer a
# midpoint or take a number from general background information.
_SELECTED_NUMBER = re.compile(
    r"(?:我(?:会)?(?:给出|估计|填|倾向)(?:为|是)?|(?:这里|因此|所以)(?:我)?(?:给出一个整数|整数回答|估计为)|"
    r"若必须填整数[，,]?\s*我填|整数回答|这里取中间值[，,]?\s*比如|"
    r"\bI(?:'d| would| will)?\s+(?:rate(?:\s+it)?|estimate|give|choose)|"
    r"\b(?:my\s+(?:score|rating|estimate)|final\s+(?:answer|score))\s*(?:is)?|"
    r"一般(?:可|可以)?给)\s*[:：=]?\s*(?:about|roughly|approximately|约|大约)?\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)(?!\d|[eE]|\.\d)", re.I)

_IOS_RELATIONS = {
    "colleague": r"\bcolleagues?\b|同事",
    "family": r"\bfamily(?:\s+members?)?\b|家庭成员|家人",
    "relative": r"\brelatives?\b|亲戚",
    "friend": r"\bfriends?\b|朋友",
}


def _ios_relations(line: str) -> set[str]:
    # Inspect the relationship label, before its answer and explanation.
    # 'relatively independent' is not a label for the relatives item.
    label = re.split(r"[:：→]|[—–]|\s+-\s+", _plain(line), maxsplit=1)[0]
    return {key for key, pattern in _IOS_RELATIONS.items() if re.search(pattern, label, re.I)}


@dataclass
class _Record:
    number: Optional[int]
    context: Optional[str]
    section: Optional[str]
    lines: list[str] = field(default_factory=list)
    value: Optional[str] = None
    segment: int = 0
    final_summary: bool = False
    sequence_list: bool = False
    identified_slot: Optional[str] = None
    explicit_number: bool = False

    @property
    def body(self):
        return "\n".join(self.lines).strip()


def _groupness_subject(slot: ScaleSlot) -> Optional[str]:
    if slot.context_id != "attribution_groupness":
        return None
    match = re.search(r"认为(?:一个|一家)?(.+?)是一个群体|perceive\s+(?:an?\s+)?(.+?)\s+to be a group", slot.question, re.I)
    return next((part for part in match.groups() if part), None) if match else None


def _contextual_lines(text: str, slots: Sequence[ScaleSlot]) -> list[str]:
    """Expose compact scenario/section headings without changing answer text."""
    lines = []
    subjects = [re.escape(label) for slot in slots if (label := _groupness_subject(slot))]
    for raw in text.splitlines():
        clean = _plain(raw).replace("<", "").replace(">", " ")
        numbered = _ITEM.match(clean)
        if numbered and _OUTER.match(numbered.group(2)):
            clean = numbered.group(2)
        outer = _OUTER.match(clean)
        if outer:
            lines.append(clean[:outer.end()])
            raw = clean[outer.end():].lstrip(" :：.、")
            raw = re.sub(r"^[-—–]\s+(?=(?:III|II|I|A|B|C)\s*[:.：、])", "", raw)
        groupness = _GROUPNESS_HEADING.match(clean)
        if groupness and groupness.group(1):
            lines.append(clean[:groupness.start(1)])
            raw = groupness.group(1)
        if subjects:
            raw = re.sub(r"[,，]\s*(?=(?:" + "|".join(subjects) + r")(?:\s|[:：]|$))", "\n", raw, flags=re.I)
        if _SECTION.match(_plain(raw)):
            raw = re.sub(r"(?:[;；]\s*|\s+)(?=(?:III|II|I|A|B|C)\s*[.:：、])", "\n", raw)
        lines.extend(raw.split("\n"))
    return lines


def _section_sequences(records: list[_Record], slots: Sequence[ScaleSlot]) -> list[_Record]:
    from .prompting import _free_positional_answers, _free_positional_tokens

    def eligible(record):
        return [s for s in slots if s.context_id == record.context
                and (record.section is None or s.section_id == record.section)]

    def validated(body, selected):
        answers = _free_positional_answers(body, selected) if selected else None
        return answers is not None and all(a.parse_status == "parsed" for a in answers)

    # A detached final six-value block can be the final group-perception task,
    # only after every scenario's A/B/C answers is complete and appears once.
    group_slots = [s for s in slots if s.context_id == "attribution_groupness"]
    if records and group_slots and not any(r.context == "attribution_groupness" for r in records):
        last = records[-1]
        parts = re.split(r"\n\s*\n", last.body)
        scenario_contexts = {s.context_id for s in slots if s.context_id != "attribution_groupness"}
        expected = {(c, section) for c in scenario_contexts for section in
                    ("attribution_agreement", "other_actor_probability", "same_actor_probability")}
        keys = [(r.context, r.section) for r in records]
        if (last.section == "same_actor_probability" and len(parts) == 2
            and len(keys) == len(expected) and set(keys) == expected
            and validated(parts[1], group_slots)
            and all(validated(parts[0] if r is last else r.body, eligible(r)) for r in records)):
            last.lines = parts[0].splitlines()
            records.append(_Record(None, "attribution_groupness", None, parts[1].splitlines(),
                                   segment=last.segment, final_summary=last.final_summary, sequence_list=True))
    expanded = []
    for record in records:
        if record.sequence_list:
            selected = eligible(record)
            body = record.body
            if str(record.context).startswith("human_rights_") and len(selected) == 2:
                pair = re.fullmatch(r"\s*A\s*[:=]\s*([+-]?\d+(?:\.\d+)?)\s*[,;]?\s*B\s*[:=]\s*([+-]?\d+(?:\.\d+)?)\s*", _plain(body), re.I)
                if pair:
                    body = " ".join(pair.groups())
            if validated(body, selected):
                for token in _free_positional_tokens(body):
                    expanded.append(_Record(None, record.context, record.section, [token], token,
                                            record.segment, record.final_summary))
            # A heading or an incomplete unlabelled list cannot supply answers.
        else:
            expanded.append(record)
    return expanded


def _records(text: str, slots: Sequence[ScaleSlot]) -> list[_Record]:
    from .prompting import (_free_split_markdown_row, _free_header_is_answer,
                            _free_table_item_number)
    contexts = {s.context_id for s in slots}
    attribution = any(str(s.slot_id).startswith("attribution_") for s in slots)
    human_rights = any(str(s.slot_id).startswith("human_rights_e") for s in slots)
    reasoning = any(str(s.slot_id).startswith("reasoning_") for s in slots)
    ios = any(str(s.slot_id).startswith("ios_") for s in slots)
    records: list[_Record] = []
    context = section = None
    current = None
    headers: list[str] = []
    segment = 0
    final_summary = False
    lines = _contextual_lines(str(text or ""), slots) if attribution or human_rights else str(text or "").splitlines()
    group_subjects = [(s, label) for s in slots if (label := _groupness_subject(s))] if attribution else []
    for raw in lines:
        line = _plain(raw)
        heading = (line.endswith((":", "：")) or raw.lstrip().startswith("#")
                   or (raw.strip().startswith("**") and raw.strip().endswith("**")))
        if not _ITEM.match(line) and "|" not in raw:
            if len(line) < 130 and _SUMMARY.search(line) and (heading or _SUMMARY.match(line)):
                segment += 1
                final_summary = True
                current = None
                headers = []
                continue
            scores_heading = bool(re.fullmatch(r"(?:scores|分数汇总|第二任务分数)\s*[:：]?", line, re.I))
            task_heading = heading or scores_heading or re.match(r"^(?:task\s*[12]|任务[一二12]|第[一二](?:任务|部分)|logical\s+validity|now\s+scoring)", line, re.I)
            if (reasoning and len(line) < 80 and task_heading
                and not re.search(r"前面|后面|我会|先说明|要求|I'll|I will|evaluate each", line, re.I)):
                if re.search(r"logical\s+validity|逻辑(?:有效性|正确性|推导|判断)|任务一|task\s*1", line, re.I):
                    section, current = "logical_validity", None
                    segment += 1
                    continue
                if scores_heading or re.search(r"real.world\s+(?:truth|knowledge)|truth\s+scor|now\s+scoring|believab|任务二|task\s*2|信念|真实性|可信度|一般(?:情况|常识).*(?:评分|分数|打分)", line, re.I):
                    section, current = "conclusion_believability", None
                    segment += 1
                    continue
        outer = _OUTER.match(line.replace("<", "").replace(">", " "))
        if outer and (attribution or human_rights):
            token = outer.group(1).casefold()
            number = int(token) if token.isdigit() else _NUMBERS.get(token)
            prefix = "attribution_s" if attribution else "human_rights_e"
            context = f"{prefix}{number:02d}" if number is not None else None
            section = None
            current = None
            headers = []
            if human_rights:
                current = _Record(None, context, None, [], segment=segment,
                                  final_summary=final_summary, sequence_list=True)
                records.append(current)
            continue
        groupness = _GROUPNESS_HEADING.match(line) if attribution else None
        if groupness:
            context, section, current = "attribution_groupness", None, None
            headers = []
            rest = groupness.group(1) or ""
            current = _Record(None, context, section, [rest], segment=segment, final_summary=final_summary, sequence_list=True)
            records.append(current)
            continue
        named_group = [(s, line[len(label):].lstrip(" :：")) for s, label in group_subjects
                       if line.casefold().startswith(label.casefold())
                       and (len(line) == len(label) or line[len(label)] in " :：")]
        if len(named_group) == 1:
            slot, tail = named_group[0]
            context, section = slot.context_id, slot.section_id
            current = _Record(None, context, section, [raw.strip()], value=tail or None,
                              segment=segment, final_summary=final_summary, identified_slot=slot.slot_id)
            records.append(current)
            continue
        sub = _SECTION.match(line)
        if sub and context in contexts:
            label = sub.group(1).upper()
            section = ({"I": "viewpoint", "II": "outcome", "III": "action"}.get(label)
                       if human_rights else {"A": "attribution_agreement", "B": "other_actor_probability", "C": "same_actor_probability"}.get(label))
            current = None
            headers = []
            if human_rights and section is not None:
                parts = re.split(r"[;；]\s*", sub.group(2))
                selected = []
                for part in parts:
                    matches = [s for s in slots if s.context_id == context and s.section_id == section
                               and (tail := _question_tail(s, _plain(part))) and _SCALAR.match(tail)]
                    selected.append(matches[0] if len(matches) == 1 else None)
                if (len(parts) == 2 and all(selected)
                    and len({s.slot_id for s in selected}) == 2):
                    # Each side repeats a different exact source statement.
                    # Semicolons in explanations alone do not establish a pair.
                    for part, slot in zip(parts, selected):
                        current = _Record(None, context, section, [part], segment=segment,
                                          final_summary=final_summary, identified_slot=slot.slot_id)
                        records.append(current)
                    continue
            if attribution and label in {"A", "B", "C"}:
                original_sub = _SECTION.match(_source_line(raw))
                current = _Record(None, context, section, [original_sub.group(2) if original_sub else raw], segment=segment, final_summary=final_summary, sequence_list=label == "A")
                records.append(current)
            elif human_rights and (re.search(r"\*\*[+-]?\d+(?:\.\d+)?\*\*\s*$", raw.strip())
                  or any(s.context_id == context and s.section_id == section
                         and (tail := _question_tail(s, _plain(sub.group(2)))) and _SCALAR.match(tail) for s in slots)):
                # Some responses repeat I/II/III on each answer, instead of
                # writing a section heading followed by two numbered rows.
                current = _Record(None, context, section, [sub.group(2)], segment=segment, final_summary=final_summary)
                records.append(current)
            elif human_rights and section is not None:
                current = _Record(None, context, section, [sub.group(2)], segment=segment,
                                  final_summary=final_summary, sequence_list=True)
                records.append(current)
            continue
        cells = _free_split_markdown_row(raw) if "|" in raw else None
        if cells:
            answer_columns = [i for i, cell in enumerate(cells) if _free_header_is_answer(cell)
                              or reasoning and re.search(r"逻辑(?:能否|判断)|logical\s+(?:validity|decision)", cell, re.I)]
            # Header cells are words, never answer values.
            if answer_columns and not any(re.search(r"\d", cell) for cell in cells):
                headers = cells
                current = None
                continue
            first = _plain(cells[0])
            numeric_id = _free_table_item_number(first)
            if re.fullmatch(r"[+-]?\d+\.\d+", first):
                continue
            indices = [i for i in range(len(cells)) if i < len(headers) and (_free_header_is_answer(headers[i])
                       or reasoning and re.search(r"逻辑(?:能否|判断)|logical\s+(?:validity|decision)", headers[i], re.I))]
            if not indices and len(cells) == 2:
                indices = [1]
            if not indices:
                continue
            value = "\n".join(cells[i] for i in indices)
            row_context, row_section, row_number = context, section, numeric_id
            if human_rights and headers and re.search(r"情境|episode|scenario", headers[0], re.I):
                scoped = re.match(r"^(III|II|I)\s*[-–]\s*([12])\b", _plain(cells[1]), re.I)
                if numeric_id is None or not scoped:
                    continue
                row_context = f"human_rights_e{numeric_id:02d}"
                row_section = {"I": "viewpoint", "II": "outcome", "III": "action"}[scoped.group(1).upper()]
                row_number = int(scoped.group(2))
            current = _Record(row_number, row_context, row_section, [" | ".join(cells)], value, segment, final_summary)
            records.append(current)
            continue
        marker = _ITEM.match(line)
        if marker:
            original_marker = _ITEM.match(_source_line(raw))
            current = _Record(int(marker.group(1)), context, section,
                              [original_marker.group(2) if original_marker else raw],
                              segment=segment, final_summary=final_summary, explicit_number=True)
            records.append(current)
            continue
        if ios and re.search(r"[:：→]|[—–]", line) and len(_ios_relations(line)) == 1:
            current = _Record(None, context, section, [raw.strip()], segment=segment, final_summary=final_summary)
            records.append(current)
            continue
        exact = ([s for s in slots
                  if (context is None or s.context_id == context
                      or attribution and context in contexts and s.context_id == "attribution_groupness")
                  and (section is None or s.section_id == section
                       or attribution and context in contexts and s.context_id == "attribution_groupness")
                  and (tail := _question_tail(s, line)) and _SCALAR.match(tail)]
                 if re.search(r"[a-zA-Z\u4e00-\u9fff]{4}", line) else [])
        if len(exact) == 1:
            slot = exact[0]
            context, section = slot.context_id, slot.section_id
            current = _Record(None, slot.context_id, slot.section_id, [raw.strip()],
                              segment=segment, final_summary=final_summary)
            records.append(current)
            continue
        # Unnumbered answer rows can still identify an exact source question.
        if (re.match(r"^\s*[-•]\s+", raw) and re.search(r"[:：→]", line)
            and not re.match(r"^(?:score|rating|评分|分数|得分|一般情况分数|逻辑(?:判断|推导)?|premises?\s*\d*|前提\s*\d*|conclusion|结论|reason|explanation)\s*[:：=]", line, re.I)):
            current = _Record(None, context, section, [raw.strip()], segment=segment, final_summary=final_summary)
            records.append(current)
            continue
        if current is not None:
            current.lines.append(raw.rstrip())
    return _section_sequences(records, slots) if attribution or human_rights else records


def _question_tail(slot: ScaleSlot, line: str) -> Optional[str]:
    question = _plain(slot.question.splitlines()[0]) if slot.question else ""
    marker = _ITEM.match(question)
    full_question = marker.group(2) if marker else question
    question = full_question.rstrip(".。?？")
    nested = _ITEM.match(line)
    if nested and nested.group(2).startswith(question):
        line = nested.group(2)
    if (slot.slot_id.startswith("attribution_s") and slot.section_id == "attribution_agreement"
        and not line.startswith(question)):
        # Observed copied wording differs only in this Chinese pronoun. Keep
        # the actor, causal attribute, event and every other character exact.
        for original, alternate in (("影响了她的行为", "影响了他的行为"),
                                    ("影响了他的行为", "影响了她的行为")):
            if original in question:
                variant = question.replace(original, alternate, 1)
                if line.startswith(variant):
                    question = variant
                    full_question = full_question.replace(original, alternate, 1)
                    break
    if len(_signature(question)) >= 4 and line.startswith(question):
        prefix = full_question if line.startswith(full_question) else question
        tail = line[len(prefix):].lstrip(" \t:：=→。?？")
        if re.match(r"^[—–][0-9]", tail):
            return tail  # Adjacent dash may be a minus; do not flip its sign.
        tail = tail.lstrip("—– ")
        tail = re.sub(r"^\.(?!\d)", "", tail, count=1).lstrip(" \t:：=→—–。?？")
        # A spaced ASCII dash after the exact source question is a separator;
        # a signed value such as -2 keeps its minus sign.
        return re.sub(r"^-\s+", "", tail, count=1)
    # Fallback: the model may abbreviate a long item by dropping parenthetical
    # examples / event clauses (e.g. "（例如他的性格、态度或气质）"). The words
    # outside parentheses still distinguish the four causal items, so align on
    # the parenthesis-stripped core and read the value after the final marker.
    if (slot.slot_id.startswith("attribution_s") and slot.section_id == "attribution_agreement"):
        base = re.sub(r"（[^（）]*）|\([^()]*\)", "", question)
        base = re.sub(r"\s+", "", base)
        core_variants = {base}
        # The model may also flip the pronoun while abbreviating (她/他).
        for old, new in (("影响了她的行为", "影响了他的行为"),
                         ("影响了他的行为", "影响了她的行为")):
            if old in base:
                core_variants.add(base.replace(old, new, 1))
        line_sig = re.sub(r"\s+", "", line)
        if len(base) >= 4 and any(line_sig.startswith(cv) for cv in core_variants):
            value_match = re.search(r"[:：=]\s*([^:：=]+?)\s*$", line)
            if value_match:
                candidate = value_match.group(1).strip()
                if re.match(r"^[—–][0-9]", candidate):
                    return candidate
                return candidate.lstrip("—– ")
    return None


def _logical_decision(line: str) -> Optional[str]:
    """Read an explicit premise-based decision, never a truth-score alias."""
    if re.search(r"\bnot\s+(?:logically\s+)?invalid\b|(?:并非|不是)\s*(?:逻辑(?:上)?无效|不能推出)|\bif\b.*\blogically\s+(?:valid|invalid)\b|如果.*逻辑.*(?:有效|无效)", line, re.I):
        return None  # Double negations and hypothetical statements need review.
    if re.match(r"^[+-]?\d", line):
        parts = re.split(r"\s+[—–]\s*", line, maxsplit=1)
        if len(parts) < 2:
            return None
        line = parts[1]
    if re.match(r"^(?:一般情况|(?:评分|分数|得分|score|rating)\s*[:：])", line, re.I):
        return None
    if re.search(r"\b(?:YES|NO|VALID|INVALID)(?:\s+(?:or|and)\s+|\s*[/／|,，]\s*|\s+)(?:YES|NO|VALID|INVALID)\b", line, re.I):
        return None
    # The surrounding wording is required: a premise beginning with 'No',
    # or a belief score labelled 'definitely true', is not a logical answer.
    negative = re.search(r"(?:\b(?:this|the\s+conclusion|conclusion|it)\s+(?:does\s+not|doesn't|cannot|can't)\s+(?:logically\s+)?follow\b|"
                         r"\b(?:not\s+logically\s+valid|logically\s+(?:not\s+valid|invalid)|(?:the\s+)?conclusion\s+contradicts\s+(?:the\s+)?(?:premises?|valid\s+inference)|this\s+does\s+not\s+imply)\b|"
                         r"(?:并非|不是|不具备|没有)\s*逻辑(?:上)?(?:的)?(?:有效|成立)|"
                         r"(?:结论|逻辑|推理|推导)[^。；\n]{0,12}(?:不能推出|不成立|无效|不有效))", line, re.I)
    positive = re.search(r"\b(?:(?:this|the\s+conclusion|conclusion|it)\s+)?follows\s+logically\s+from\s+(?:the\s+)?premises\b|"
                         r"\b(?:this|the\s+conclusion|conclusion|it)\s+follows\s+(?:logically|from\s+(?:the\s+)?premises)\b|\blogically\s+valid\b|"
                         r"(?:结论|逻辑|推理|推导)[^。；\n]{0,12}(?:成立|有效)", line, re.I)
    decisions = set()
    if negative:
        decisions.add("NO")
    if positive and not negative:
        decisions.add("YES")
    for match in re.finditer(r"(?:^|[。.!:：—–]\s*|\s)(YES|NO|VALID|INVALID)\s*(?=$|[.!:：,(（]|from\s+(?:the\s+)?premises)", line, re.I):
        if re.search(r"(?:not\s+(?:logically\s+)?|logically\s+not\s+)$", line[:match.start(1)], re.I):
            continue
        decisions.add("YES" if match.group(1).upper() in {"YES", "VALID"} else "NO")
    marked = re.search(r"(?:逻辑(?:判断|推导|判断结果)?|已判断|判断|答案|answer)\s*[:：]\s*(不能推出|能推出|可推出|是|否|YES|NO)(?=$|[。，,.!（(\s])", line, re.I)
    if marked:
        decisions.add("YES" if marked.group(1).upper() in {"是", "YES", "能推出", "可推出"} else "NO")
    standalone = re.search(r"(?:^|[。；]|所以|因此)\s*(不能推出|能推出|可推出)(?=$|[。，！])", line)
    if standalone:
        decisions.add("NO" if standalone.group(1) == "不能推出" else "YES")
    leading = re.match(r"^(是|否|YES|NO)(?=$|[。.!（(])", line, re.I)
    if leading:
        decisions.add("YES" if leading.group(1).upper() in {"是", "YES"} else "NO")
    return next(iter(decisions)) if len(decisions) == 1 else None


def _decode(slot: ScaleSlot, body: str, value: Optional[str] = None):
    from .prompting import (ParsedAnswer, _free_parse_value, _free_extract_numeric,
                           _free_numeric_answer, _free_text_response, _is_open_text_slot,
                           _free_extract_choice, _free_extract_ios_pair, _FREE_SCORE_MARKER,
                           _RANGE_ALNUM, _RANGE_OR, _RANGE_BETWEEN,
                           _free_extract_interval, _free_range_answer, _free_source_option)
    if _is_open_text_slot(slot):
        return _free_text_response(slot, body)
    target = value if value is not None else body
    if (not slot.choices and len(set(re.findall(r"[①②③④⑤⑥⑦⑧⑨⑩]", target))) >= 3
        and len(re.findall(r"[①②③④⑤⑥⑦⑧⑨⑩][^\n]*?[:：=]", target)) >= 3):
        return ParsedAnswer(answer=body, slot_id=slot.slot_id, parse_status="option_codebook",
                            parse_error="The response lists scores for several options without selecting one answer.")
    lines = [_plain(line) for line in target.splitlines() if line.strip()]
    candidates = []
    resolutions = []
    ranges = []
    original_lines = [line.strip() for line in target.splitlines() if line.strip()]
    for position, line in enumerate(lines):
        interval = None
        if not line or line == "---":
            continue
        if len(re.findall(r"(?<!\d)\d+(?:\.\d+)?\s*=", line)) >= 3:
            # Repeated response-option definitions are not a selected score.
            continue
        if re.match(r"^[+-]?\d+(?:\.\d+)?\s*[,;]\s*[+-]?\d", line):
            # A compact aggregate score list cannot become this one item's score.
            continue
        if value is None and position == 0:
            tail = _question_tail(slot, line)
            if tail is not None:
                if re.match(r"^[—–][0-9]", tail):
                    return ParsedAnswer(answer=body, slot_id=slot.slot_id, parse_status="unparsed",
                                        parse_error="An adjacent dash may denote a negative value or a separator; no sign was guessed.")
                if not tail:
                    continue
                if re.match(r"^(?:[+-]?(?:\d|\.\d)|[①②③④⑤⑥⑦⑧⑨⑩]|YES\b|NO\b|VALID\b|INVALID\b|[AB](?=$|[.：:（(\s]))", tail, re.I):
                    line = tail
        source_option = _free_source_option(slot, line)
        if source_option is not None:
            candidates.append(_free_parse_value(slot, source_option))
            continue
        if slot.choices:
            if slot.response_type == "ios_pair":
                chosen = _free_extract_ios_pair(line, slot.choices)
                if chosen is not None:
                    candidates.append(_free_parse_value(slot, chosen))
                continue
            choice_line = re.split(r"\s+[—–]|\bbecause\b|因为|理由[:：]", line, maxsplit=1, flags=re.I)[0].strip()
            if "→" in choice_line:
                choice_line = choice_line.rsplit("→", 1)[-1].strip()
            chosen = _free_extract_choice(choice_line, slot.choices)
            if slot.section_id == "logical_validity":
                chosen = _logical_decision(line)
            elif (all(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", c) for c in slot.choices)
                  and not _ITEM.match(choice_line)):
                numeric_target = _FREE_SCORE_MARKER.sub("", choice_line, count=1).strip()
                interval = _free_extract_interval(numeric_target)
                if interval is not None:
                    ranges.append(_free_range_answer(slot, body, interval))
                    continue
                numeric, numeric_status = _free_extract_numeric(numeric_target)
                if numeric_status == "parsed":
                    token = str(int(numeric)) if numeric.is_integer() else str(numeric)
                    if token in slot.choices:
                        chosen = token
                    else:
                        return ParsedAnswer(answer=body, slot_id=slot.slot_id, parse_status="invalid_choice",
                                            parse_error=f"The explicit value {token} is not among the source choices; it was not replaced.")
            if chosen is not None:
                candidates.append(_free_parse_value(slot, chosen))
                continue
            continue
        numeric, status = None, "unparsed"
        answer_target = None
        numeric_line = re.sub(r"^(?:约|大约|about|approximately|around)\s*", "", line, flags=re.I)
        circled = re.search(r"(?:^|[:：→—–]\s*)(?:[*_`\s]*)([①②③④⑤⑥⑦⑧⑨⑩])", original_lines[position])
        if circled:
            if len(set(re.findall(r"[①②③④⑤⑥⑦⑧⑨⑩]", original_lines[position]))) > 1:
                return ParsedAnswer(answer=body, slot_id=slot.slot_id, parse_status="conflicting_values", parse_error="Several circled options were listed without one selection.")
            numeric, status = float("①②③④⑤⑥⑦⑧⑨⑩".index(circled.group(1)) + 1), "parsed"
        elif (_SCALAR.match(numeric_line) or _free_extract_interval(numeric_line) is not None
              or any(pattern.match(numeric_line) for pattern in (_RANGE_ALNUM, _RANGE_OR, _RANGE_BETWEEN))):
            answer_target = numeric_line
        elif position == 0 and _question_tail(slot, _plain(original_lines[position])) is not None and re.search(r"\*\*([+-]?\d+(?:\.\d+)?)\*\*\s*$", original_lines[position]):
            answer_target = re.search(r"\*\*([+-]?\d+(?:\.\d+)?)\*\*\s*$", original_lines[position]).group(1)
        elif _SELECTED_NUMBER.search(line):
            selected = _SELECTED_NUMBER.search(line)
            answer_target = line[selected.start("value"):]
        elif (re.match(r"^(?:一般情况分数|分数|评分|score|rating)\s*[:：]", line, re.I)
              and re.search(r"\*\*[+-]?\d+(?:\.\d+)?(?:\s*[（(][^\d()（）]*[）)])?\*\*", original_lines[position])):
            emphasized = re.findall(r"\*\*([+-]?\d+(?:\.\d+)?)(?:\s*[（(][^\d()（）]*[）)])?\*\*", original_lines[position])
            if len(set(emphasized)) == 1:
                number_match = re.search(r"\*\*" + re.escape(emphasized[0]), original_lines[position])
                answer_target = _plain(original_lines[position][number_match.start():])
        elif _FREE_SCORE_MARKER.search(line):
            marker = _FREE_SCORE_MARKER.search(line)
            answer_target = line[marker.end():].strip()
        elif re.search(r"(?:score|rating|评分|分数|得分|分值|一般情况|概率|可能性|likelihood|probability|justification|rightness)\s*[:：=]|(?:评分|分数)(?:为|是)", line, re.I):
            suffix = re.split(r"[:：=]|(?:评分|分数)(?:为|是)", line, maxsplit=1)[-1].strip()
            suffix = re.sub(r"^(?:约|大约|about|approximately|around)\s*", "", suffix, flags=re.I)
            answer_target = suffix
        elif (position == 0 or "→" in line) and re.search(r"[:：→]|[—–]", line):
            separators = re.finditer(r"(?:[:：→]|[—–]{1,2})\s*(?=(?:约\s*)?[+-]?(?:\d|\.\d))", line)
            separator = next((match for match in separators
                              if not re.search(r"\d\s*[%％]?\s*$", line[:match.start()])), None)
            suffix = line[separator.end():].strip() if separator else ""
            suffix = re.sub(r"^(?:约|大约|about|approximately|around)\s*", "", suffix, flags=re.I)
            if _SCALAR.match(suffix):
                answer_target = suffix
        if answer_target is not None:
            # Only the selected answer prefix supplies a range, not a rationale.
            numeric, status = _free_extract_numeric(answer_target)
            interval = _free_extract_interval(answer_target)
            if interval is not None:
                numeric, status = None, "range_unresolved"
        # A score written as 2/7 is not a range when 7 is this item's scale maximum.
        fraction = re.search(r"(?:^|[:：→])\s*(?:约)?\s*([+-]?\d+(?:\.\d+)?)\s*/\s*(\d+)\s*(?:[（(]|$)", line)
        if fraction and slot.maximum is not None and float(fraction.group(2)) == slot.maximum:
            numeric, status = float(fraction.group(1)), "parsed"
        if numeric is not None:
            candidates.append(_free_numeric_answer(slot, body, numeric, status))
            if _SELECTED_NUMBER.search(line) or re.search(r"最终(?:答案|分数)", line):
                resolutions.append(numeric)
        elif status == "range_unresolved":
            ranges.append(_free_range_answer(slot, body, interval) if interval is not None else None)
    keys = {(str(a.answer) if slot.choices else a.score) for a in candidates}
    if len(keys) > 1:
        return ParsedAnswer(answer=body, slot_id=slot.slot_id, parse_status="conflicting_values", parse_error="Several distinct answers were supplied for this item.")
    if ranges and not (len(keys) == 1 and resolutions and set(resolutions) == keys):
        if (not candidates and all(a is not None for a in ranges)
            and len({(a.range_lower, a.range_upper, a.range_unit) for a in ranges}) == 1):
            return ranges[0]
        return ParsedAnswer(answer=body, slot_id=slot.slot_id, parse_status="range_unresolved", parse_error="A range or alternatives were retained without selecting one value.")
    if candidates:
        answer = candidates[0]
        if not slot.choices:
            answer.answer = body
        return answer
    if slot.response_type == "text" and body.strip():
        return _free_text_response(slot, body)
    return ParsedAnswer(answer=body or None, slot_id=slot.slot_id, parse_status="unparsed", parse_error="No explicit answer of the expected type was found.")


def _task_subject(slot: ScaleSlot, line: str) -> bool:
    """Recognise short labels in the four source probability-estimation tasks."""
    if slot.response_type != "number" or slot.maximum != 100 or slot.context_id != "items":
        return False
    question = _plain(slot.question)
    for pattern in (r"分手|break\s*up", r"幼儿园|kindergarten", r"富有|become\s+rich|becoming\s+rich", r"象棋|棋手|chess|输给|输棋"):
        if re.search(pattern, question, re.I) and re.search(pattern, line, re.I):
            return True
    return False


def _reasoning_landmarks(slots: Sequence[ScaleSlot]) -> dict[str, set[str]]:
    """Unique source-conclusion words identify terse, paraphrased English rows."""
    common = {"some", "have", "with", "from", "that", "this", "those", "their", "they",
              "does", "things", "living", "animals", "individuals", "country", "member",
              "good", "would", "could", "should", "will", "been"}
    words = {}
    for slot in slots:
        if slot.section_id != "logical_validity":
            continue
        conclusion = re.search(r"Conclusion\s*[:：]\s*([^\n]+)", slot.question, re.I)
        if conclusion:
            words[slot.slot_id] = set(re.findall(r"[a-z]{4,}", conclusion.group(1).casefold())) - common
    return {key: {word for word in vocabulary if sum(word in other for other in words.values()) == 1}
            for key, vocabulary in words.items()}


def _reasoning_typed_records(records: list[_Record], slots: Sequence[ScaleSlot]) -> list[_Record]:
    """Separate bare numbered decisions and scores before resolving item numbers.

    Original question numbers restart in the belief task. The answer type
    identifies its task, while the saved source/display order still determines
    which question it belongs to. Combined prose is left to the normal decoder.
    """
    sections = {slot.section_id for slot in slots if slot.slot_id.startswith("reasoning_")}
    if not {"logical_validity", "conclusion_believability"}.issubset(sections):
        return records
    inferred_numeric = defaultdict(list)
    for record in records:
        if record.section is not None or record.number is None:
            continue
        body = _plain(record.value if record.value is not None else record.body)
        if re.fullmatch(r"(?:是|否|YES|NO|VALID|INVALID|正确|错误)[.!。！]?", body, re.I):
            record.section = "logical_validity"
        elif re.fullmatch(r"(?:(?:分数|评分|score|rating)\s*[:：=]\s*)?[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?[。.!]?", body, re.I):
            record.section = "conclusion_believability"
            inferred_numeric[record.segment].append(record)
    belief_numbers = set()
    for slot in slots:
        if slot.section_id == "conclusion_believability":
            marker = _ITEM.match(_plain(slot.question.splitlines()[0]))
            number = int(marker.group(1)) if marker else slot.metadata.get("argument_no")
            if number is not None:
                belief_numbers.add(int(number))
    for group in inferred_numeric.values():
        if belief_numbers and any(record.number not in belief_numbers for record in group):
            # A scored 24-item logic list is not the separate 16-item task.
            # Do not salvage a few coincidentally equal display/source IDs.
            for record in group:
                record.context = "unresolved_reasoning_numeric_task"
    return records


def decode_free_response(text: str, slots: Sequence[ScaleSlot], *, allow_positional_fallback=False):
    from .prompting import (ParsedAnswer, ParseResult, FREE_STATUS_READABLE,
                           _parse_free_answers_legacy, _is_open_text_slot, _free_text_response,
                           _free_positional_answers, FREE_PARSER_VERSION)
    if len(slots) == 1 and _is_open_text_slot(slots[0]) and str(text or "").strip():
        answer = _free_text_response(slots[0], str(text).strip())
        answer.mapping_method = "single_open_item"
        return ParseResult([answer], "ok", recovery=FREE_PARSER_VERSION)
    if allow_positional_fallback:
        positional = _free_positional_answers(text, slots)
        if positional is not None:
            for answer in positional:
                answer.mapping_method = "complete_response_order"
            read = sum(a.parse_status in FREE_STATUS_READABLE for a in positional)
            return ParseResult(positional, "ok" if read == len(slots) else "partial" if read else "unparsed",
                               recovery=f"{FREE_PARSER_VERSION}-positional-v1")
        # Uniform bare sequence: every token is the same readable value. Order
        # is irrelevant (all identical), so this is unambiguous even without
        # item numbers; tolerate at most two duplicated tokens.
        uni_tokens = re.findall(r"[+-]?[0-9A-Za-z\u4e00-\u9fff]+", str(text))
        if uni_tokens:
            sample = uni_tokens[0]
            if (all(tok == sample for tok in uni_tokens)
                    and len(slots) <= len(uni_tokens) <= len(slots) + 2
                    and all(_decode(s, sample).parse_status in FREE_STATUS_READABLE for s in slots)):
                uniform_answers = []
                for s in slots:
                    dec = _decode(s, sample)
                    dec.mapping_method = "uniform_bare_sequence"
                    uniform_answers.append(dec)
                return ParseResult(uniform_answers, "ok", recovery=FREE_PARSER_VERSION)
        reasoning_sections = {s.section_id for s in slots if s.slot_id.startswith("reasoning_")}
        if {"logical_validity", "conclusion_believability"}.issubset(reasoning_sections):
            # Type-segregated recovery: a bare response may list the 24 yes/no
            # logical decisions and the 16 numeric belief scores as two
            # unlabelled blocks. Assign each token to the single task section
            # that can read it; recover only when both tasks are exact/complete.
            task_pairs = [(task, [i for i, s in enumerate(slots) if s.section_id == task])
                          for task in ("logical_validity", "conclusion_believability")]
            bare_tokens = [t.strip() for t in str(text).splitlines() if t.strip()]
            # Valid only for fully unlabelled tokens; explicit numbers make the
            # source-vs-display mapping ambiguous because the model may renumber.
            has_item_numbers = any(_ITEM.match(_plain(t)) for t in bare_tokens)
            assigned = {task: [] for task, _ in task_pairs}
            usable = True
            for tok in bare_tokens:
                owners = [task for task, idx in task_pairs
                          if any(_decode(slots[i], tok).parse_status in FREE_STATUS_READABLE for i in idx)]
                if len(owners) != 1:
                    usable = False
                    break
                assigned[owners[0]].append(tok)
            if usable and not has_item_numbers and all(assigned[t] for t, _ in task_pairs):
                both = [ParsedAnswer(slot_id=s.slot_id, parse_status="missing") for s in slots]
                recovered_tasks = 0
                for task, idx in task_pairs:
                    seq = assigned[task]
                    decoded_pairs = [(j, _decode(slots[j], tok)) for j, tok in zip(idx, seq)]
                    if (len(seq) == len(idx)
                            and all(dec.parse_status in FREE_STATUS_READABLE for _, dec in decoded_pairs)):
                        for j, dec in decoded_pairs:
                            dec.mapping_method = "typed_task_blocks"
                            both[j] = dec
                        recovered_tasks += 1
                if recovered_tasks:
                    read = sum(a.parse_status in FREE_STATUS_READABLE for a in both)
                    if read == len(slots):
                        return ParseResult(both, "ok", recovery=FREE_PARSER_VERSION)
                    return ParseResult(both, "partial", recovery=FREE_PARSER_VERSION,
                                       error="Only complete, unambiguous typed answer blocks were recovered; the other block was incomplete/mismatched and raw_response was retained.")
            for task in ("logical_validity", "conclusion_believability"):
                task_indices = [i for i, s in enumerate(slots) if s.section_id == task]
                task_answers = _free_positional_answers(text, [slots[i] for i in task_indices])
                if task_answers is not None and all(a.parse_status == "parsed" for a in task_answers):
                    answers = [ParsedAnswer(slot_id=s.slot_id, parse_status="missing") for s in slots]
                    for index, answer in zip(task_indices, task_answers):
                        answer.mapping_method = "complete_typed_task_order"
                        answers[index] = answer
                    return ParseResult(answers, "partial", recovery=FREE_PARSER_VERSION,
                                       error="Only one complete typed task was supplied; the other task remains missing.")
    records = _records(text, slots)
    if not records:
        return _parse_free_answers_legacy(text, slots, allow_positional_fallback=allow_positional_fallback)
    records = _reasoning_typed_records(records, slots)
    answers = [ParsedAnswer(slot_id=s.slot_id, parse_status="missing") for s in slots]
    groups = defaultdict(list)
    landmarks = _reasoning_landmarks(slots)
    for record in records:
        groups[(record.context, record.section, record.segment)].append(record)
    for (context, section, _segment), group in groups.items():
        eligible = [i for i, s in enumerate(slots) if (context is None or s.context_id == context) and (section is None or s.section_id == section)]
        if not eligible:
            continue
        source_numbers = {}
        for i in eligible:
            match = _ITEM.match(_plain(slots[i].question.splitlines()[0]))
            number = int(match.group(1)) if match else slots[i].metadata.get("argument_no")
            if number is not None:
                source_numbers.setdefault(int(number), []).append(i)
        mapped = []
        mapping_methods = []
        paired_matches = []
        for record in group:
            first = _signature(record.lines[0]) if record.lines else ""
            conclusion = re.search(r"(?:Conclusion|结论)\s*[:：]\s*([^\n]+)", record.body, re.I)
            stated_conclusion = _signature(conclusion.group(1)) if conclusion else ""
            label = _signature(re.split(r"[:：→]|[—–]", _plain(record.lines[0]), maxsplit=1)[0]) if record.lines else ""
            subject_words = set(re.findall(r"[a-z]{4,}", _plain(record.lines[0]).casefold())) if record.lines else set()
            subject_families = {key for key, words in landmarks.items() if words & subject_words}
            matches = []
            prefixes = []
            full_questions = {}
            aliases = []
            for i in eligible:
                source = slots[i].question
                if slots[i].section_id == "logical_validity":
                    conclusion = re.search(r"(?:Conclusion|结论)\s*[:：]\s*([^\n]+)", source, re.I)
                    source = conclusion.group(1) if conclusion else source
                q = _signature(next(iter(source.splitlines()), ""))
                # Exact question wording is evidence, unlike fuzzy paraphrase matching.
                if (len(q) >= (2 if re.search(r"[\u4e00-\u9fff]", q) else 4)
                    and (q in first or q in stated_conclusion)):
                    matches.append(i)
                    full_questions[i] = q
                elif len(first) >= 20 and first in q:
                    matches.append(i)
                elif len(q) >= 24:
                    length = 14 if re.search(r"[\u4e00-\u9fff]", q) else 32
                    if len(q) >= length and q[:length] in first:
                        prefixes.append(i)
                if slots[i].slot_id.startswith("ios_"):
                    if slots[i].slot_id[4:] in _ios_relations(record.lines[0]):
                        aliases.append(i)
                if _task_subject(slots[i], _plain(record.lines[0])):
                    aliases.append(i)
                family = slots[i].linked_slot_id or slots[i].slot_id
                if len(subject_families) == 1 and family in subject_families:
                    aliases.append(i)
                # A short, unique source subject such as '特朗普：4' can be
                # mapped without relying on the respondent's item numbering.
                minimum = 2 if re.search(r"[\u4e00-\u9fff]", label) else 8
                if (minimum <= len(label) <= 30 and label in q
                    and re.search(r"[:：→]|[—–]", _plain(record.lines[0]))):
                    aliases.append(i)
            # A longer complete question can contain a different, shorter
            # question (e.g. "How often do you pray?"). Prefer the longer
            # wording only when every competing wording is nested within it.
            if full_questions:
                longest = max(full_questions, key=lambda i: len(full_questions[i]))
                winners = [i for i, q in full_questions.items() if q == full_questions[longest]]
                if len(winners) == 1 and all(q in full_questions[longest] for q in full_questions.values()):
                    matches = [i for i in matches if i not in full_questions or i == longest]
            # Relationship/subject labels distinguish questions whose wording
            # begins with the same generic sentence.
            overlap = list(set(matches).intersection(aliases))
            if len(matches) > 1 and len(overlap) == 1:
                matches = overlap
            method = "question_text" if matches else "task_subject" if aliases else "question_prefix"
            matches = list(dict.fromkeys(matches or aliases or prefixes))
            if record.identified_slot:
                matches = [i for i in eligible if slots[i].slot_id == record.identified_slot]
                method = "source_subject"
            pair = []
            if len(matches) > 1:
                typed = [i for i in matches if _decode(slots[i], record.body, record.value).parse_status in FREE_STATUS_READABLE]
                if (len(typed) == 2
                    and {slots[i].section_id for i in typed} == {"logical_validity", "conclusion_believability"}
                    and any(slots[i].linked_slot_id == slots[j].slot_id for i, j in (typed, typed[::-1]))):
                    pair = typed
                matches = typed if len(typed) == 1 else matches
            mapped.append(matches[0] if len(matches) == 1 else None)
            mapping_methods.append(method)
            paired_matches.append(pair)
        votes = set()
        for record, index, pair in zip(group, mapped, paired_matches):
            if index is None and pair:
                index = pair[0]
            if index is None or record.number is None:
                continue
            original = source_numbers.get(record.number, [])
            display = eligible[record.number - 1] if 1 <= record.number <= len(eligible) else None
            if index in original and index != display:
                votes.add("source")
            elif index == display and index not in original:
                votes.add("display")
        labels = [record.number for record in group]
        source_sequence = [next((n for n, ids in source_numbers.items() if ids == [i]), None) for i in eligible]
        source_mode = votes == {"source"} or (not votes and labels == source_sequence and len(set(labels)) == len(labels) and labels != list(range(1, len(eligible) + 1)))
        positional_table = (len(group) == len(eligible) and all(r.number is None for r in group)
                            and (context is not None or not source_numbers)
                            and all(index is None or index == eligible[position] for position, index in enumerate(mapped))
                            and all(_decode(slots[i], r.body, r.value).parse_status in FREE_STATUS_READABLE
                                    for i, r in zip(eligible, group)))
        for position, (record, index) in enumerate(zip(group, mapped)):
            method = mapping_methods[position]
            if index is None and record.number is not None:
                display = eligible[record.number - 1] if 1 <= record.number <= len(eligible) else None
                if source_mode and len(source_numbers.get(record.number, [])) == 1:
                    index, method = source_numbers[record.number][0], "source_number"
                elif source_mode and len(source_numbers.get(record.number, [])) == 2:
                    typed = [i for i in source_numbers[record.number]
                             if _decode(slots[i], record.body, record.value).parse_status in FREE_STATUS_READABLE]
                    if len(typed) == 1:
                        index, method = typed[0], "source_number"
                    elif (len(typed) == 2 and any(slots[i].linked_slot_id == slots[j].slot_id for i, j in (typed, typed[::-1]))):
                        paired_matches[position] = typed
                elif display is not None and source_numbers.get(record.number) == [display]:
                    index, method = display, "aligned_number"
                elif display is not None and (not source_numbers or votes == {"display"}):
                    index, method = display, "local_number"
            if index is None and positional_table:
                index, method = eligible[position], "complete_section_order"
            if index is None and record.number is None and len(eligible) == 1:
                index, method = eligible[0], "single_section_item"
            targets = paired_matches[position] or ([index] if index is not None else [])
            if paired_matches[position]:
                method = "linked_task_question"
            for index in targets:
                answer = _decode(slots[index], record.body, record.value)
                answer.mapping_method = method
                previous = answers[index]
                if previous.parse_status == "missing":
                    answers[index] = answer
                elif previous.parse_status == "conflicting_answers":
                    previous.answer = str(previous.answer) + "\n\n" + record.body
                elif _is_open_text_slot(slots[index]) and answer.parse_status == "text_response":
                    previous.answer = str(previous.answer) + "\n\n" + record.body
                elif (previous.parse_status in {"range_response", "range_out_of_range", "range_unresolved"}
                      and answer.score is not None and record.final_summary):
                    answer.answer = str(previous.answer) + "\n\n" + record.body
                    answer.parse_error = "The model supplied an explicit, source-identified value in its final summary after a range."
                    answers[index] = answer
                elif previous.parse_status in FREE_STATUS_READABLE and answer.parse_status in FREE_STATUS_READABLE:
                    if (previous.score != answer.score
                        or (previous.range_lower, previous.range_upper, previous.range_unit)
                        != (answer.range_lower, answer.range_upper, answer.range_unit)
                        or (slots[index].choices and previous.range_text is None and previous.answer != answer.answer)):
                        answers[index] = ParsedAnswer(answer=str(previous.answer) + "\n\n" + record.body, slot_id=slots[index].slot_id, parse_status="conflicting_answers", parse_error="Repeated answers conflict; neither was selected.", mapping_method=method)
                    elif previous.range_text is not None:
                        previous.answer = str(previous.answer) + "\n\n" + record.body
                elif previous.parse_status == "unparsed" and answer.parse_status in FREE_STATUS_READABLE:
                    answers[index] = answer
                elif answer.parse_status == "unparsed" and previous.parse_status in FREE_STATUS_READABLE:
                    if not slots[index].choices:
                        previous.answer = str(previous.answer) + "\n\n" + record.body
                else:
                    # An unresolved repeated answer must not disappear merely
                    # because another occurrence supplied a readable value.
                    answers[index] = ParsedAnswer(answer=str(previous.answer) + "\n\n" + record.body,
                                                   slot_id=slots[index].slot_id, parse_status="conflicting_answers",
                                                   parse_error="Repeated answers include an unresolved value; raw text was retained.",
                                                   mapping_method=method)
        # With no wording evidence, shuffled source numbers and display numbers
        # can both be plausible. Recover only values identical under BOTH maps.
        complete_labels = list(range(1, len(eligible) + 1))
        if (not votes and not source_mode and len(group) == len(eligible)
            and all(n is not None for n in labels) and sorted(labels) == complete_labels
            and sorted(source_numbers) == complete_labels
            and all(len(ids) == 1 for ids in source_numbers.values())):
            by_number = {r.number: r for r in group}
            for position, index in enumerate(eligible, 1):
                if answers[index].parse_status != "missing" or _is_open_text_slot(slots[index]):
                    continue
                original_number = next(n for n, ids in source_numbers.items() if ids == [index])
                source_record, display_record = by_number[original_number], by_number[position]
                a, b = (_decode(slots[index], r.body, r.value) for r in (source_record, display_record))
                same_value = (a.score, a.range_lower, a.range_upper, a.range_unit) == (b.score, b.range_lower, b.range_upper, b.range_unit)
                if (a.parse_status == b.parse_status and a.parse_status in FREE_STATUS_READABLE
                    and same_value and (a.score is not None or a.range_lower is not None
                                        or slots[index].choices and a.answer == b.answer)):
                    a.mapping_method = "numbering_invariant_value"
                    if not slots[index].choices and source_record.body != display_record.body:
                        a.answer = source_record.body + "\n\n" + display_record.body
                    answers[index] = a
    read = sum(a.parse_status in FREE_STATUS_READABLE for a in answers)
    status = "ok" if read == len(slots) else "partial" if read else "unparsed"
    return ParseResult(answers=answers, status=status, recovery=FREE_PARSER_VERSION, error=None if status == "ok" else "Some answers could not be mapped or decoded unambiguously; raw_response was retained.")


def decode_interview_rows(text: str, slots: Sequence[ScaleSlot]):
    """Locate source interview rows by their unique first follow-up question.

    A respondent may merge several continuation rows into one named story.
    Their story numbers must not be mistaken for the workbook's row indices.
    """
    from .prompting import ParsedAnswer, ParseResult, FREE_PARSER_VERSION
    lines = str(text or "").splitlines()
    numbered = _complete_interview_numbering(lines, slots)
    if numbered is not None:
        complete = all(a.parse_status == "text_response" for a in numbered)
        return ParseResult(numbered, "ok" if complete else "partial",
                           error=None if complete else "Only a complete prefix of interview rows was supplied; remaining rows were retained as missing.",
                           recovery=f"interview-question-order-{FREE_PARSER_VERSION}")
    anchors = {}
    first_questions = {}
    for index, slot in enumerate(slots):
        questions = [line for line in slot.question.splitlines() if _ITEM.match(_plain(line))]
        if questions:
            first_questions[index] = _signature(questions[0])
    prefix_anchors = set()
    for index, slot in enumerate(slots):
        if index not in first_questions:
            continue
        key = first_questions[index]
        if len(key) < 8:
            continue
        positions = [position for position, line in enumerate(lines)
                     if _ITEM.match(_plain(line)) and key in _signature(line)]
        if len(positions) == 1:
            anchors[index] = positions[0]
        elif not positions:
            length = 14 if re.search(r"[\u4e00-\u9fff]", key) else 32
            prefix = key[:length]
            if len(prefix) < length or sum(q.startswith(prefix) for q in first_questions.values()) != 1:
                continue
            positions = [position for position, line in enumerate(lines)
                         if _ITEM.match(_plain(line)) and _signature(line).startswith(prefix)]
            if len(positions) == 1:
                anchors[index] = positions[0]
                prefix_anchors.add(index)
    if not anchors or len(set(anchors.values())) != len(anchors):
        return None
    answers = [ParsedAnswer(slot_id=s.slot_id, parse_status="missing") for s in slots]
    ordered = sorted(anchors.items(), key=lambda pair: pair[1])
    for position, (index, start) in enumerate(ordered):
        end = ordered[position + 1][1] if position + 1 < len(ordered) else len(lines)
        # Stop before an explicit next-story header, not inside its subquestions.
        for boundary in range(start + 1, end):
            if _OUTER.match(_plain(lines[boundary])):
                end = boundary
                break
        body = "\n".join(lines[start:end]).strip()
        answers[index] = ParsedAnswer(answer=body, score=None, slot_id=slots[index].slot_id,
                                      parse_status="text_response", mapping_method=("interview_question_prefix" if index in prefix_anchors else "interview_source_question"))
    read = sum(a.parse_status == "text_response" for a in answers)
    return ParseResult(answers, "ok" if read == len(slots) else "partial",
                       error=None if read == len(slots) else "Some source interview rows lack unique question boundaries; raw_response was retained.",
                       recovery=f"interview-source-rows-{FREE_PARSER_VERSION}")


def _complete_interview_numbering(lines: list[str], slots: Sequence[ScaleSlot]):
    """Segment fixed-order interview rows only with a complete number skeleton."""
    from .prompting import ParsedAnswer, _is_open_text_slot
    marker = re.compile(r"^(\d+)[a-z]?\s*[.、):：]\s*(.*)$", re.I)

    def runs(source):
        found = []
        for position, line in enumerate(source):
            match = marker.match(_plain(line))
            if match and (not found or found[-1][0] != int(match.group(1))):
                found.append((int(match.group(1)), position))
        return found

    if len(slots) < 2 or not all(_is_open_text_slot(s) for s in slots):
        return None
    source_runs = [runs(s.question.splitlines()) for s in slots]
    if any(not row for row in source_runs):
        return None
    expected = [number for row in source_runs for number, _ in row]
    actual = runs(lines)
    boundaries, total = [], 0
    for row in source_runs:
        total += len(row)
        boundaries.append(total)
    if (len(actual) < 4 or len(actual) not in boundaries[1:]
        or [number for number, _ in actual] != expected[:len(actual)]):
        return None
    starts = {actual[i][1] for i in [0] + boundaries[:-1] if i < len(actual)}
    # Story headings may merge continuation rows. Each supplied heading must
    # still precede an actual source-row boundary, never interrupt its questions.
    headings = [position for position, line in enumerate(lines) if _OUTER.match(_plain(line))]
    for heading in headings:
        following = next((position for _, position in actual if position > heading), None)
        if following not in starts:
            return None
    answers, cursor = [], 0
    for index, (slot, row) in enumerate(zip(slots, source_runs)):
        if cursor == len(actual):
            answers.append(ParsedAnswer(slot_id=slot.slot_id, parse_status="missing"))
            continue
        start = actual[cursor][1]
        previous = actual[cursor - 1][1] if cursor else -1
        start = next((h for h in headings if previous < h < start), start)
        cursor += len(row)
        end = actual[cursor][1] if cursor < len(actual) else len(lines)
        end = next((h for h in headings if actual[cursor - 1][1] < h < end), end)
        body = "\n".join(lines[start:end]).strip()
        if not re.search(r"[a-zA-Z\u4e00-\u9fff]", re.sub(r"(?m)^\s*\d+[a-z]?\s*[.、):：]", "", body)):
            return None
        answers.append(ParsedAnswer(answer=body, slot_id=slot.slot_id, score=None,
                                    parse_status="text_response", mapping_method="complete_interview_question_order"))
    return answers
