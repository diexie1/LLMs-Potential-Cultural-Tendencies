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
_SCALAR = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*[%％分人]|\s*/\s*\d+)?(?:\s*(?:[=:(（—–,;。；-]|[.!?]\s*$|$))")
_SUMMARY = re.compile(r"summary|summari[sz]|总结|汇总|整理|合成.*答案|最终(?:答案|分数|回答)|final\s+(?:answers?|scores?|list)|百分比列表", re.I)

# These identify a value the respondent explicitly selected, including a
# final scalar supplied after discussing an interval. They do not infer a
# midpoint or take a number from general background information.
_SELECTED_NUMBER = re.compile(
    r"(?:我(?:会)?(?:给出|估计|填|倾向)(?:为|是)?|(?:这里|因此|所以)(?:我)?(?:给出一个整数|整数回答|估计为)|"
    r"若必须填整数[，,]?\s*我填|整数回答|这里取中间值[，,]?\s*比如|"
    r"\bI(?:'d| would| will)?\s+(?:rate(?:\s+it)?|estimate|give|choose)|"
    r"\b(?:my\s+(?:score|rating|estimate)|final\s+(?:answer|score))\s*(?:is)?|"
    r"一般(?:可|可以)?给)\s*[:：=]?\s*(?:about|roughly|approximately|约|大约)?\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+))(?!\d|\.\d)", re.I)

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

    @property
    def body(self):
        return "\n".join(self.lines).strip()


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
    for raw in str(text or "").splitlines():
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
            task_heading = heading or re.match(r"^(?:task\s*[12]|任务[一二12]|第[一二](?:任务|部分)|logical\s+validity|now\s+scoring)", line, re.I)
            if (reasoning and len(line) < 80 and task_heading
                and not re.search(r"前面|后面|我会|先说明|要求|I'll|I will|evaluate each", line, re.I)):
                if re.search(r"logical\s+validity|逻辑(?:有效性|正确性|推导|判断)|任务一|task\s*1", line, re.I):
                    section, current = "logical_validity", None
                    segment += 1
                    continue
                if re.search(r"real.world\s+(?:truth|knowledge)|truth\s+scor|now\s+scoring|believab|任务二|task\s*2|信念|真实性|可信度|一般(?:情况|常识).*(?:评分|分数|打分)", line, re.I):
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
            continue
        if attribution and re.search(r"group(?:ness|\s+perception)|群体(?:实体性|感知|知觉|性|评分)", line, re.I) and not _ITEM.match(line):
            context, section, current = "attribution_groupness", None, None
            headers = []
            continue
        sub = _SECTION.match(line)
        if sub and context in contexts:
            label = sub.group(1).upper()
            section = ({"I": "viewpoint", "II": "outcome", "III": "action"}.get(label)
                       if human_rights else {"A": "attribution_agreement", "B": "other_actor_probability", "C": "same_actor_probability"}.get(label))
            current = None
            headers = []
            if attribution and label in {"B", "C"}:
                original_sub = _SECTION.match(_source_line(raw))
                current = _Record(None, context, section, [original_sub.group(2) if original_sub else raw], segment=segment, final_summary=final_summary)
                records.append(current)
            elif human_rights and re.search(r"\*\*[+-]?\d+(?:\.\d+)?\*\*\s*$", raw.strip()):
                # Some responses repeat I/II/III on each answer, instead of
                # writing a section heading followed by two numbered rows.
                current = _Record(None, context, section, [sub.group(2)], segment=segment, final_summary=final_summary)
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
                              [original_marker.group(2) if original_marker else raw], segment=segment, final_summary=final_summary)
            records.append(current)
            continue
        if ios and re.search(r"[:：→]|[—–]", line) and len(_ios_relations(line)) == 1:
            current = _Record(None, context, section, [raw.strip()], segment=segment, final_summary=final_summary)
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
    return records


def _question_tail(slot: ScaleSlot, line: str) -> Optional[str]:
    question = _plain(slot.question.splitlines()[0]) if slot.question else ""
    marker = _ITEM.match(question)
    question = marker.group(2) if marker else question
    nested = _ITEM.match(line)
    if nested and nested.group(2).startswith(question):
        line = nested.group(2)
    if len(_signature(question)) >= 4 and line.startswith(question):
        return line[len(question):].lstrip(" \t:：=→—–")
    return None


def _logical_decision(line: str) -> Optional[str]:
    """Read an explicit premise-based decision, never a truth-score alias."""
    if re.match(r"^[+-]?\d", line):
        parts = re.split(r"\s+[—–]\s*", line, maxsplit=1)
        if len(parts) < 2:
            return None
        line = parts[1]
    if re.match(r"^(?:一般情况|(?:评分|分数|得分|score|rating)\s*[:：])", line, re.I):
        return None
    if re.search(r"\b(?:YES|NO|VALID|INVALID)\s*(?:or|and)\s*(?:YES|NO|VALID|INVALID)\b", line, re.I):
        return None
    # The surrounding wording is required: a premise beginning with 'No',
    # or a belief score labelled 'definitely true', is not a logical answer.
    negative = re.search(r"(?:\b(?:this|the\s+conclusion|conclusion|it)\s+(?:does\s+not|doesn't|cannot|can't)\s+(?:logically\s+)?follow\b|"
                         r"\b(?:logically\s+invalid|(?:the\s+)?conclusion\s+contradicts\s+(?:the\s+)?(?:premises?|valid\s+inference)|this\s+does\s+not\s+imply)\b|"
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
                           _free_extract_interval, _free_range_answer)
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
        if re.match(r"^[+-]?\d+(?:\.\d+)?\s*[,;]\s*[+-]?\d", line):
            # A compact aggregate score list cannot become this one item's score.
            continue
        if value is None and position == 0:
            tail = _question_tail(slot, line)
            if tail is not None:
                if not tail:
                    continue
                if re.match(r"^(?:[+-]?\d|[①②③④⑤⑥⑦⑧⑨⑩]|YES\b|NO\b|VALID\b|INVALID\b|[AB](?=$|[.：:（(\s]))", tail, re.I):
                    line = tail
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
        if slot.response_type == "integer" and answer.score is not None and not float(answer.score).is_integer():
            return ParsedAnswer(answer=body, slot_id=slot.slot_id, parse_status="non_integer", parse_error="The supplied value is not an integer; it was not rounded.")
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


def decode_free_response(text: str, slots: Sequence[ScaleSlot], *, allow_positional_fallback=False):
    from .prompting import (ParsedAnswer, ParseResult, FREE_STATUS_READABLE,
                           _parse_free_answers_legacy, _is_open_text_slot, _free_text_response,
                           FREE_PARSER_VERSION)
    if len(slots) == 1 and _is_open_text_slot(slots[0]) and str(text or "").strip():
        answer = _free_text_response(slots[0], str(text).strip())
        answer.mapping_method = "single_open_item"
        return ParseResult([answer], "ok", recovery=FREE_PARSER_VERSION)
    records = _records(text, slots)
    if not records:
        return _parse_free_answers_legacy(text, slots, allow_positional_fallback=allow_positional_fallback)
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
            if index is None and len(eligible) == 1:
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
