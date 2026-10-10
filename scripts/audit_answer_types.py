"""Offline per-item type/option audit against the currently loaded workbooks."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.prompting import FREE_PARSER_VERSION, _free_parse_value, parse_free_answers
from app.response_decoder import _decode
from app.scale_loader import discover_scales


def examples(slot):
    if slot.choices:
        return [(c, float(c) if c.lstrip('+-').isdigit() else None,
                 'parsed', c) for c in slot.choices]
    if slot.response_type == 'text':
        return [('This is my explanation. 这是我的回答。', None, 'text_response', None)]
    low, high = slot.minimum, slot.maximum
    values = sorted({low, high, int((low + high) / 2)})
    if slot.response_type == 'number':
        values.append(low + .5)
    return [(str(v), float(v), 'parsed', None) for v in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    scales = discover_scales(args.source)
    if not scales:
        parser.error('No source workbooks found')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be new or empty')
    args.output.mkdir(parents=True, exist_ok=True)
    rows, errors, groups = [], [], []
    totals = Counter(scales=len(scales))
    types = Counter()
    for scale in scales:
        for language in ('ch', 'en'):
            sheet = getattr(scale, language)
            if sheet is None:
                errors.append(f'{scale.name}/{language}: missing sheet')
                continue
            totals['sheets'] += 1
            if sheet.profile_error or not sheet.is_profiled:
                errors.append(f'{scale.name}/{language}: missing/invalid profile')
            json_values = []
            expected_values = []
            for slot in sheet.slots:
                failures, checks = [], 0
                cases = examples(slot)
                aliases = slot.metadata.get('answer_labels', {})
                for label, code in aliases.items():
                    score = float(code) if code.lstrip('+-').isdigit() else None
                    cases.append((label, score, 'parsed', code if slot.choices else None))
                    cases.append(('选择：' + label, score, 'parsed', code if slot.choices else None))
                for value, score, status, choice in cases:
                    for name, decoder in (('value', _free_parse_value), ('body', _decode)):
                        answer = decoder(slot, value)
                        checks += 1
                        if (answer.score != score or answer.parse_status != status
                            or choice is not None and answer.answer != choice):
                            failures.append({'path': name, 'input': value, 'expected_score': score,
                                             'actual_score': answer.score, 'actual_status': answer.parse_status})
                if slot.response_type in {'integer', 'number'}:
                    interval = _decode(slot, f'{slot.minimum}–{slot.maximum}')
                    checks += 1
                    if interval.score is not None or interval.range_lower != slot.minimum or interval.range_upper != slot.maximum:
                        failures.append({'path': 'interval', 'status': interval.parse_status})
                if slot.response_type == 'integer':
                    for decoder in (_free_parse_value, _decode):
                        answer = decoder(slot, str(slot.minimum + .5))
                        checks += 1
                        if answer.score is not None or answer.parse_status != 'non_integer':
                            failures.append({'path': 'non_integer', 'status': answer.parse_status})
                initial = examples(slot)[0]
                json_values.append({'display_index': len(json_values) + 1, 'answer': initial[0]})
                expected_values.append(initial)
                totals['items'] += 1
                totals['checks'] += checks
                types[slot.response_type] += 1
                row = {'scale': scale.name, 'language': language, 'slot_id': slot.slot_id,
                       'question': slot.question, 'response_type': slot.response_type,
                       'minimum': slot.minimum, 'maximum': slot.maximum,
                       'choices': json.dumps(slot.choices, ensure_ascii=False),
                       'source_option_labels': json.dumps(aliases, ensure_ascii=False),
                       'checks': checks, 'passed': not failures,
                       'limitations': '按Excel情境行保存；行内追问不拆字段' if slot.response_type == 'text' else '',
                       'failures': json.dumps(failures, ensure_ascii=False)}
                rows.append(row)
                if failures:
                    errors.append({key: row[key] for key in ('scale', 'language', 'slot_id', 'failures')})
            result = parse_free_answers(json.dumps({'answers': json_values}, ensure_ascii=False), sheet.slots)
            for answer, expected in zip(result.answers, expected_values):
                totals['checks'] += 1
                if answer.score != expected[1] or answer.parse_status != expected[2] or expected[3] is not None and answer.answer != expected[3]:
                    errors.append(f'{scale.name}/{language}/{answer.slot_id}: full JSON mapping failed')
            groups.append({'scale': scale.name, 'language': language, 'profile': sheet.profile_id,
                           'items': len(sheet.slots), 'types': dict(Counter(s.response_type for s in sheet.slots))})
    with (args.output / '逐题答案类型核验.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'parser_version': FREE_PARSER_VERSION, 'source': str(args.source.resolve()),
               'totals': dict(totals), 'types': dict(types), 'groups': groups, 'errors': errors,
               'scope': 'Known canonical values and exact source labels; does not prove arbitrary prose is correctly mapped.'}
    (args.output / '答案类型核验汇总.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'totals': dict(totals), 'types': dict(types), 'errors': errors}, ensure_ascii=False))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
