"""构建 Prompt，解析回答，并映射回原始题目顺序。"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .scale_loader import ScaleSheet


def build_prompt(sheet: ScaleSheet, order: Sequence[int], language: str = "") -> str:
    """
    原封不动拼接表格内容（不添加任何自定义提示词）：
    A2 + \\n + 打乱后的题目（A3/A4/...），题目之间以换行符分隔。
    """
    parts: List[str] = [sheet.instruction]
    for orig_i in order:
        parts.append(sheet.questions[orig_i])
    return "\n".join(parts)



def parse_scores(text: str, n_items: int) -> List[Optional[float]]:
    """从模型输出中解析 n_items 个分数，按展示顺序返回。"""
    scores: List[Optional[float]] = [None] * n_items

    # 优先尝试 JSON
    json_match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            if isinstance(obj, dict):
                for k, v in obj.items():
                    m = re.search(r"\d+", str(k))
                    if not m:
                        continue
                    idx = int(m.group()) - 1
                    if 0 <= idx < n_items:
                        scores[idx] = float(v)
                if any(s is not None for s in scores):
                    return scores
            if isinstance(obj, list) and len(obj) >= n_items:
                return [float(x) for x in obj[:n_items]]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # 行内 "1: 7" / "1. 7" / "Item 1 = 7"
    pattern = re.compile(
        r"(?:^|\n)\s*(?:Item|题目|Q)?\s*(\d+)\s*[:：\.\)、\-]\s*(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n_items:
            scores[idx] = float(m.group(2))

    if sum(1 for s in scores if s is not None) >= max(1, n_items // 2):
        return scores

    # 兜底：按顺序提取纯数字
    nums = re.findall(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])", text)
    if len(nums) >= n_items:
        return [float(x) for x in nums[:n_items]]

    return scores


def shuffle_order(n: int, rng: Optional[random.Random] = None) -> List[int]:
    order = list(range(n))
    (rng or random).shuffle(order)
    return order


def remap_to_original(
    shuffled_scores: Sequence[Optional[float]],
    order: Sequence[int],
) -> List[Optional[float]]:
    """将打乱题目上的分数映射回原始题号顺序。"""
    result: List[Optional[float]] = [None] * len(order)
    for display_pos, orig_i in enumerate(order):
        if display_pos < len(shuffled_scores):
            result[orig_i] = shuffled_scores[display_pos]
    return result


def build_result_payload(
    *,
    scale_name: str,
    language: str,
    provider: str,
    model: str,
    temperature: float,
    order_id: int,
    seed: int,
    order: Sequence[int],
    sheet: ScaleSheet,
    scores_original: Sequence[Optional[float]],
    raw_response: str,
    prompt: str,
) -> Dict[str, Any]:
    items = []
    for i, q in enumerate(sheet.questions):
        items.append(
            {
                "index": i + 1,
                "question": q,
                "score": scores_original[i],
            }
        )
    return {
        "scale_name": scale_name,
        "scale_title": sheet.title,
        "language": language,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "order_id": order_id,
        "shuffle_seed": seed,
        "shuffle_order": [i + 1 for i in order],  # 1-based 展示用
        "instruction": sheet.instruction,
        "items": items,
        "scores": list(scores_original),
        "raw_response": raw_response,
        "prompt": prompt,
    }
