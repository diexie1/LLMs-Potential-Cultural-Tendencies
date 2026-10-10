"""Reparse retained model responses offline into a separate output directory."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.prompting import (FREE_PARSER_VERSION, FREE_STATUS_READABLE, parse_free_answers,
                          parse_kohlberg_free_answers, classify_model_response, remap_to_original)
from app.scale_loader import ScaleSlot, attach_source_answer_labels
from app.runner import BatchRunner
from app.trial_results import export_trials


def reparse_result(value: dict) -> dict:
    """Use the saved schema and permutation, without reopening modified workbooks."""
    result = copy.deepcopy(value)
    items = result["items"]
    fields = ScaleSlot.__dataclass_fields__
    canonical = [ScaleSlot(**{key: item[key] for key in fields if key in item}) for item in items]
    attach_source_answer_labels(canonical, result.get("instruction", ""))
    by_slot = {slot.slot_id: slot for slot in canonical}
    for block in result.get("blocks", []):
        if block.get("instruction"):
            attach_source_answer_labels([by_slot[item["slot_id"]] for item in block["slots"]], block["instruction"])
    order = [int(index) - 1 for index in result.get("shuffle_order", range(1, len(items) + 1))]
    if sorted(order) != list(range(len(items))):
        raise ValueError("Saved shuffle order is not a complete permutation")
    slots = [canonical[index] for index in order]
    raw = result.get("raw_response", "")
    parsed = (parse_kohlberg_free_answers(raw, slots)
              if result.get("scale_profile", {}).get("id") == "kohlberg_mji_v1"
              else parse_free_answers(raw, slots, allow_positional_fallback=True))
    answers = remap_to_original(parsed.answers, order)
    for item, answer in zip(items, answers):
        item.update(answer=answer.answer, score=answer.score, parse_status=answer.parse_status,
                    range_text=answer.range_text, range_lower=answer.range_lower,
                    range_upper=answer.range_upper, range_unit=answer.range_unit,
                    parse_error=answer.parse_error, mapping_method=answer.mapping_method,
                    answer_present=answer.answer is not None and bool(str(answer.answer).strip()),
                    score_present=answer.score is not None)
    result["response_parser"] = {"version": FREE_PARSER_VERSION, "status": parsed.status,
                                 "error": parsed.error, "recovery": parsed.recovery}
    result["parser_version"] = FREE_PARSER_VERSION
    result["response_status"] = classify_model_response(raw, parsed)
    result["scores"] = [item["score"] for item in items]
    result["answered_item_count"] = sum(item["answer_present"] for item in items)
    result["read_item_count"] = sum(item["parse_status"] in FREE_STATUS_READABLE for item in items)
    result["scored_item_count"] = sum(item["score_present"] for item in items)
    result["all_items_read"] = result["read_item_count"] == len(items)
    by_id = {item["slot_id"]: item for item in items}
    for block in result.get("blocks", []):
        block["slots"] = [copy.deepcopy(by_id[item["slot_id"]]) for item in block["slots"]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--skip-excel", action="store_true", help="Write trial JSON/CSV and audit summaries without the full-response workbook")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_dir():
        parser.error("Source directory does not exist")
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Source and output must be separate, non-overlapping directories")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be new or empty; existing files will not be overwritten")
    files = sorted(source.rglob("order_*.json"))
    if not files:
        parser.error("No retained trial JSON files found")
    output.mkdir(parents=True, exist_ok=True)
    audit = []
    totals = Counter()
    groups = defaultdict(Counter)
    item_statuses = Counter()
    item_changes = []
    for path in files:
        saved_bytes = path.read_bytes()
        old = json.loads(saved_bytes.decode("utf-8"))
        result = reparse_result(old)
        result["reparse_audit"] = {"source_file": path.relative_to(source).as_posix(),
                                  "source_sha256": hashlib.sha256(saved_bytes).hexdigest(),
                                  "previous_parser": old.get("parser_version"),
                                  "previous_status": old.get("response_status")}
        # Original text, request and ordering must survive offline decoding unchanged.
        for key in ("raw_response", "prompt", "shuffle_order", "request_snapshot", "response_snapshot"):
            if result.get(key) != old.get(key):
                raise RuntimeError(f"Retained source field changed: {key}")
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        BatchRunner._write_csv(target.with_suffix(".csv"), result)
        row = {"scale": old["scale_name"], "language": old["language"], "trial": old["order_id"],
               "before_status": old.get("response_status"), "after_status": result["response_status"],
               "before_read": old.get("read_item_count", 0), "after_read": result["read_item_count"],
               "items": len(result["items"]), "after_scored": result["scored_item_count"],
               "source": path.relative_to(source).as_posix(),
               "answer_types": ",".join(sorted({item["response_type"] for item in result["items"]})),
               "unresolved_reasons": json.dumps(dict(Counter(item["parse_status"] for item in result["items"]
                                                              if item["parse_status"] not in FREE_STATUS_READABLE or item["parse_status"] in {"out_of_range", "range_out_of_range"})), ensure_ascii=False)}
        audit.append(row)
        totals["trials"] += 1
        totals["before_" + str(row["before_status"])] += 1
        totals["after_" + row["after_status"]] += 1
        for prefix, data in (("before", old), ("after", result)):
            groups[(old["scale_name"], old["language"])][prefix + "_" + str(data.get("response_status"))] += 1
        totals["answer_assignment_changes"] += sum(
            before.get("score") != after.get("score") for before, after in zip(old["items"], result["items"]))
        for before, after in zip(old["items"], result["items"]):
            if before.get("score") is None and after.get("score") is not None:
                totals["scores_recovered"] += 1
            if before.get("score") is not None and before["score"] != after.get("score"):
                totals["existing_scores_changed_or_removed"] += 1
            fields = ("score", "parse_status", "range_lower", "range_upper", "range_unit")
            if any(before.get(key) != after.get(key) for key in fields):
                item_changes.append({"source": path.relative_to(source).as_posix(), "slot_id": after["slot_id"],
                                     "before_score": before.get("score"), "after_score": after.get("score"),
                                     "before_status": before.get("parse_status"), "after_status": after.get("parse_status"),
                                     "mapping_method": after.get("mapping_method"), "answer": after.get("answer")})
        totals["before_read_items"] += old.get("read_item_count", 0)
        totals["after_read_items"] += result["read_item_count"]
        totals["items"] += len(result["items"])
        item_statuses.update(item["parse_status"] for item in result["items"])
        if path.read_bytes() != saved_bytes:
            raise RuntimeError("Source file changed during offline reparse")
    for name, rows in (("逐试次解析核对.csv", audit),
                       ("待人工核对.csv", [row for row in audit if row["after_status"] != "ok" or row["unresolved_reasons"] != "{}"])):
        with (output / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
            writer.writeheader()
            writer.writerows(rows)
    if item_changes:
        with (output / "逐题解析变更.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(item_changes[0]))
            writer.writeheader()
            writer.writerows(item_changes)
    totals.setdefault("existing_scores_changed_or_removed", 0)
    summary = {"parser_version": FREE_PARSER_VERSION, "totals": dict(totals),
               "item_statuses": dict(item_statuses),
               "groups": [{"scale": scale, "language": language, **counts} for (scale, language), counts in sorted(groups.items())]}
    (output / "解析汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # Accept a collection of runs, a single run, or a single scale directory.
    # Directly stored JSON files must never be treated as export directories.
    folders = set()
    for path in files:
        parts = path.relative_to(source).parts
        folders.add(output / parts[0] if len(parts) > 1 and parts[0].startswith("run_") else output)
    if not args.skip_excel:
        for folder in sorted(folders):
            export_trials(folder)
    print(json.dumps(totals, ensure_ascii=True))


if __name__ == "__main__":
    main()
