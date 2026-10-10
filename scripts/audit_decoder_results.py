"""Compare decoder versions offline without duplicating retained trial files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.prompting import FREE_PARSER_VERSION, FREE_STATUS_READABLE
from scripts.reparse_results import reparse_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        parser.error('Source/output must not overlap')
    if output.exists() and any(output.iterdir()):
        parser.error('Output must be new or empty')
    files = sorted(p for p in source.rglob('order_*.json')
                   if re.fullmatch(r'order_\d{4}_temperature_.+\.json', p.name))
    if not files:
        parser.error('No retained trial files')
    output.mkdir(parents=True, exist_ok=True)
    totals, statuses = Counter(), Counter()
    groups = defaultdict(Counter)
    changes, trials, errors, changed_scores = [], [], [], []
    hashes = hashlib.sha256()
    for number, path in enumerate(files, 1):
        raw_bytes = path.read_bytes()
        old = json.loads(raw_bytes.decode('utf-8'))
        new = reparse_result(old)
        relative = path.relative_to(source).as_posix()
        hashes.update(relative.encode('utf-8') + b'\0' + hashlib.sha256(raw_bytes).digest())
        for key in ('raw_response', 'prompt', 'shuffle_order', 'request_snapshot', 'response_snapshot'):
            if old.get(key) != new.get(key):
                errors.append(f'{relative}: retained field changed: {key}')
        for before, after in zip(old['items'], new['items']):
            score, kind, status = after['score'], after['response_type'], after['parse_status']
            statuses[status] += 1
            if score is not None:
                invalid = (not math.isfinite(score)
                           or kind == 'text'
                           or kind == 'integer' and not float(score).is_integer()
                           or kind in {'choice', 'ios_pair'} and str(int(score) if float(score).is_integer() else score) not in after['choices'])
                if invalid:
                    errors.append(f'{relative}/{after["slot_id"]}: invalid typed score {score}')
            if before.get('score') != score:
                totals['score_changes'] += 1
                if before.get('score') is None:
                    totals['scores_recovered'] += 1
                else:
                    totals['existing_scores_changed_or_removed'] += 1
                    changed_scores.append({'source': relative, 'slot_id': after['slot_id'],
                                           'question': after['question'], 'before': before, 'after': after,
                                           'raw_response': old['raw_response']})
            fields = ('score', 'parse_status', 'range_lower', 'range_upper', 'range_unit')
            if any(before.get(key) != after.get(key) for key in fields):
                changes.append({'source': relative, 'slot_id': after['slot_id'],
                                'before_score': before.get('score'), 'after_score': score,
                                'before_status': before.get('parse_status'), 'after_status': status,
                                'question': after['question'], 'answer': after.get('answer'),
                                'mapping_method': after.get('mapping_method')})
        totals['trials'] += 1
        totals['items'] += len(new['items'])
        totals['before_read_items'] += old['read_item_count']
        totals['after_read_items'] += new['read_item_count']
        totals['before_' + old['response_status']] += 1
        totals['after_' + new['response_status']] += 1
        groups[(new['scale_name'], new['language'])]['items'] += len(new['items'])
        groups[(new['scale_name'], new['language'])]['read'] += new['read_item_count']
        trials.append({'source': relative, 'before_status': old['response_status'],
                       'after_status': new['response_status'], 'before_read': old['read_item_count'],
                       'after_read': new['read_item_count'],
                       'unresolved': json.dumps(dict(Counter(i['parse_status'] for i in new['items']
                                                           if i['parse_status'] not in FREE_STATUS_READABLE)), ensure_ascii=False)})
        if path.read_bytes() != raw_bytes:
            errors.append(f'{relative}: source bytes changed')
        if number % 500 == 0:
            print(f'{number}/{len(files)}', flush=True)
    for name, rows in (('逐题解析变更.csv', changes), ('逐试次解析核对.csv', trials)):
        if rows:
            with (output / name).open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    totals.setdefault('existing_scores_changed_or_removed', 0)
    summary = {'parser_version': FREE_PARSER_VERSION, 'source': str(source), 'totals': dict(totals),
               'item_statuses': dict(statuses), 'errors': errors, 'source_collection_sha256': hashes.hexdigest(),
               'groups': [{'scale': scale, 'language': language, **counts} for (scale, language), counts in sorted(groups.items())]}
    (output / '解析核验汇总.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / '已有分数变化核验.json').write_text(json.dumps(changed_scores, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'totals': dict(totals), 'statuses': dict(statuses), 'errors': errors}, ensure_ascii=False))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
