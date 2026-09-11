"""从 xlsx 加载中英文量表。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from openpyxl import load_workbook

from .config import DATA_DIR


@dataclass
class ScaleSheet:
    language: str  # "en" | "ch"
    title: str
    instruction: str
    questions: List[str] = field(default_factory=list)

    @property
    def n_items(self) -> int:
        return len(self.questions)


@dataclass
class ScaleFile:
    path: Path
    name: str
    en: Optional[ScaleSheet] = None
    ch: Optional[ScaleSheet] = None


def _read_column_a(ws) -> List[str]:
    values: List[str] = []
    for row in ws.iter_rows(min_col=1, max_col=1, values_only=True):
        cell = row[0]
        if cell is None:
            values.append("")
        else:
            values.append(str(cell).strip())
    # 去掉尾部空行
    while values and not values[-1]:
        values.pop()
    return values


def _parse_sheet(ws, language: str) -> Optional[ScaleSheet]:
    cells = _read_column_a(ws)
    if len(cells) < 3:
        return None
    title = cells[0] if cells[0] else "Untitled"
    instruction = cells[1] if len(cells) > 1 else ""
    questions = [q for q in cells[2:] if q]
    if not questions:
        return None
    return ScaleSheet(
        language=language,
        title=title,
        instruction=instruction,
        questions=questions,
    )


def load_scale_file(path: Path) -> ScaleFile:
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        name = path.stem
        en = ch = None
        # Sheet1 = 英文, Sheet2 = 中文
        if "Sheet1" in wb.sheetnames:
            en = _parse_sheet(wb["Sheet1"], "en")
        if "Sheet2" in wb.sheetnames:
            ch = _parse_sheet(wb["Sheet2"], "ch")
        # 兜底：按顺序取前两张
        if en is None and wb.sheetnames:
            en = _parse_sheet(wb[wb.sheetnames[0]], "en")
        if ch is None and len(wb.sheetnames) > 1:
            ch = _parse_sheet(wb[wb.sheetnames[1]], "ch")
        return ScaleFile(path=path, name=name, en=en, ch=ch)
    finally:
        wb.close()


def discover_scales(data_dir: Optional[Path] = None) -> List[ScaleFile]:
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    if not data_dir.exists():
        return []
    scales: List[ScaleFile] = []
    for path in sorted(data_dir.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        try:
            scales.append(load_scale_file(path))
        except Exception as exc:
            print(f"[warn] 跳过无法读取的文件 {path.name}: {exc}")
    return scales
