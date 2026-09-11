"""前端视觉主题（仅 UI，不影响后端逻辑）。"""

from __future__ import annotations

import platform
import tkinter as tk
from tkinter import ttk


# 清爽科研工具风：冷灰底 + 青绿强调（避免常见 AI 紫/奶油模板）
COLORS = {
    "bg": "#EEF2F6",
    "surface": "#FFFFFF",
    "surface_alt": "#F8FAFC",
    "border": "#D7DEE8",
    "border_soft": "#E6ECF3",
    "text": "#1E293B",
    "text_secondary": "#64748B",
    "text_muted": "#94A3B8",
    "accent": "#0F766E",
    "accent_hover": "#0D9488",
    "accent_soft": "#CCFBF1",
    "primary": "#0F766E",
    "primary_text": "#FFFFFF",
    "danger": "#B91C1C",
    "danger_soft": "#FEE2E2",
    "ok": "#15803D",
    "ok_soft": "#DCFCE7",
    "warn": "#B45309",
    "warn_soft": "#FEF3C7",
    "header_bg": "#0B3B3A",
    "header_text": "#F8FAFC",
    "header_sub": "#99F6E4",
    "log_bg": "#0F172A",
    "log_fg": "#E2E8F0",
    "tree_sel": "#99F6E4",
    "tree_sel_fg": "#134E4A",
}


def ui_font(size: int = 10, weight: str = "normal") -> tuple:
    system = platform.system()
    if system == "Darwin":
        family = "PingFang SC"
    elif system == "Windows":
        family = "Microsoft YaHei UI"
    else:
        family = "Noto Sans CJK SC"
    return (family, size, weight) if weight != "normal" else (family, size)


def apply_theme(root: tk.Tk) -> ttk.Style:
    root.configure(bg=COLORS["bg"])
    try:
        root.option_add("*Font", ui_font(10))
    except Exception:
        pass

    style = ttk.Style(root)
    # clam 最容易定制跨平台外观
    try:
        style.theme_use("clam")
    except Exception:
        pass

    c = COLORS
    style.configure(".", background=c["bg"], foreground=c["text"], font=ui_font(10))
    style.configure("TFrame", background=c["bg"])
    style.configure("Surface.TFrame", background=c["surface"])
    style.configure("Header.TFrame", background=c["header_bg"])
    style.configure("Card.TFrame", background=c["surface"])

    style.configure(
        "Card.TLabelframe",
        background=c["surface"],
        foreground=c["text"],
        bordercolor=c["border"],
        relief="solid",
        borderwidth=1,
    )
    style.configure(
        "Card.TLabelframe.Label",
        background=c["surface"],
        foreground=c["accent"],
        font=ui_font(10, "bold"),
    )

    style.configure(
        "Title.TLabel",
        background=c["header_bg"],
        foreground=c["header_text"],
        font=ui_font(16, "bold"),
    )
    style.configure(
        "Subtitle.TLabel",
        background=c["header_bg"],
        foreground=c["header_sub"],
        font=ui_font(9),
    )
    style.configure("TLabel", background=c["bg"], foreground=c["text"], font=ui_font(10))
    style.configure(
        "Card.TLabel",
        background=c["surface"],
        foreground=c["text"],
        font=ui_font(10),
    )
    style.configure(
        "Muted.TLabel",
        background=c["surface"],
        foreground=c["text_secondary"],
        font=ui_font(9),
    )
    style.configure(
        "Hint.TLabel",
        background=c["surface"],
        foreground=c["text_muted"],
        font=ui_font(9),
    )
    style.configure(
        "Status.TLabel",
        background=c["accent_soft"],
        foreground=c["accent"],
        font=ui_font(9),
        padding=(8, 4),
    )
    style.configure(
        "StatusProxy.TLabel",
        background=c["warn_soft"],
        foreground=c["warn"],
        font=ui_font(9),
        padding=(8, 4),
    )

    style.configure(
        "TEntry",
        fieldbackground=c["surface_alt"],
        foreground=c["text"],
        bordercolor=c["border"],
        lightcolor=c["border"],
        darkcolor=c["border"],
        insertcolor=c["text"],
        padding=6,
    )
    style.map("TEntry", bordercolor=[("focus", c["accent"])])

    style.configure(
        "TCombobox",
        fieldbackground=c["surface_alt"],
        background=c["surface"],
        foreground=c["text"],
        arrowcolor=c["accent"],
        bordercolor=c["border"],
        padding=4,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", c["surface_alt"])],
        selectbackground=[("readonly", c["accent_soft"])],
        selectforeground=[("readonly", c["text"])],
    )

    style.configure(
        "TSpinbox",
        fieldbackground=c["surface_alt"],
        foreground=c["text"],
        bordercolor=c["border"],
        arrowcolor=c["accent"],
        padding=4,
    )

    style.configure(
        "TButton",
        background=c["surface_alt"],
        foreground=c["text"],
        bordercolor=c["border"],
        focuscolor=c["accent_soft"],
        font=ui_font(9),
        padding=(12, 6),
    )
    style.map(
        "TButton",
        background=[("active", c["border_soft"]), ("pressed", c["border"])],
        foreground=[("disabled", c["text_muted"])],
    )

    style.configure(
        "Accent.TButton",
        background=c["accent"],
        foreground=c["primary_text"],
        bordercolor=c["accent"],
        font=ui_font(10, "bold"),
        padding=(16, 8),
    )
    style.map(
        "Accent.TButton",
        background=[("active", c["accent_hover"]), ("disabled", c["border"])],
        foreground=[("disabled", c["text_muted"])],
    )

    style.configure(
        "Primary.TButton",
        background=c["primary"],
        foreground=c["primary_text"],
        bordercolor=c["primary"],
        font=ui_font(10, "bold"),
        padding=(18, 9),
    )
    style.map(
        "Primary.TButton",
        background=[("active", c["accent_hover"]), ("disabled", "#94A3B8")],
    )

    style.configure(
        "Danger.TButton",
        background=c["danger_soft"],
        foreground=c["danger"],
        bordercolor=c["danger"],
        font=ui_font(10, "bold"),
        padding=(16, 8),
    )
    style.map(
        "Danger.TButton",
        background=[("active", "#FECACA"), ("disabled", c["border_soft"])],
    )

    style.configure(
        "Ghost.TButton",
        background=c["surface"],
        foreground=c["accent"],
        bordercolor=c["accent"],
        font=ui_font(9),
        padding=(10, 5),
    )

    style.configure(
        "Treeview",
        background=c["surface"],
        fieldbackground=c["surface"],
        foreground=c["text"],
        bordercolor=c["border"],
        rowheight=28,
        font=ui_font(9),
    )
    style.configure(
        "Treeview.Heading",
        background=c["surface_alt"],
        foreground=c["text_secondary"],
        relief="flat",
        font=ui_font(9, "bold"),
        bordercolor=c["border"],
    )
    style.map(
        "Treeview",
        background=[("selected", c["tree_sel"])],
        foreground=[("selected", c["tree_sel_fg"])],
    )
    style.map("Treeview.Heading", background=[("active", c["accent_soft"])])

    style.configure(
        "Horizontal.TProgressbar",
        troughcolor=c["border_soft"],
        background=c["accent"],
        bordercolor=c["border_soft"],
        lightcolor=c["accent"],
        darkcolor=c["accent"],
        thickness=10,
    )

    style.configure(
        "TScrollbar",
        background=c["border"],
        troughcolor=c["surface_alt"],
        bordercolor=c["surface_alt"],
        arrowcolor=c["text_secondary"],
    )

    style.configure("Footer.TFrame", background=c["surface"])
    style.configure(
        "Footer.TLabel",
        background=c["surface"],
        foreground=c["text_secondary"],
        font=ui_font(9),
    )

    return style


def make_card(parent: tk.Misc, title: str) -> ttk.LabelFrame:
    box = ttk.LabelFrame(parent, text=f"  {title}  ", style="Card.TLabelframe", padding=12)
    return box


def style_log_text(widget: tk.Text) -> None:
    widget.configure(
        bg=COLORS["log_bg"],
        fg=COLORS["log_fg"],
        insertbackground=COLORS["log_fg"],
        selectbackground=COLORS["accent"],
        selectforeground="#FFFFFF",
        relief="flat",
        borderwidth=0,
        padx=10,
        pady=8,
        font=ui_font(9),
        highlightthickness=1,
        highlightbackground=COLORS["border"],
        highlightcolor=COLORS["accent"],
    )
