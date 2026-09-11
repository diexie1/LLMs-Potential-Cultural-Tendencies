"""跨平台桌面界面（tkinter，Windows / macOS 均可运行）。"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

from . import proxy_util
from .api_client import list_models
from .config import (
    DEFAULT_CALL_COUNT,
    DEFAULT_CONCURRENCY,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TEMPERATURE,
    LOGS_DIR,
    get_api_credentials,
    load_user_config,
    resolve_path,
    save_user_config,
    to_rel_path,
)
from .net_info import detect_public_ip_info, network_status_text
from .runner import BatchRunner, RunnerConfig, ScaleRunConfig
from .scale_loader import ScaleFile, discover_scales
from .ui_theme import apply_theme, make_card, style_log_text

PROVIDERS = ("deepseek", "openai", "qwen", "gemini")


class ScaleApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("量表 API 批量作答工具")
        self.geometry("1120x820")
        self.minsize(1000, 720)
        apply_theme(self)

        self.scales: List[ScaleFile] = []
        self.data_dir = tk.StringVar(value=DEFAULT_DATA_DIR)
        self.results_dir = tk.StringVar(value=DEFAULT_RESULTS_DIR)
        self.provider = tk.StringVar(value="deepseek")
        self.model = tk.StringVar(value="")
        self.temperature = tk.DoubleVar(value=DEFAULT_TEMPERATURE)
        self.concurrency = tk.IntVar(value=DEFAULT_CONCURRENCY)
        self.batch_en = tk.IntVar(value=DEFAULT_CALL_COUNT)
        self.batch_ch = tk.IntVar(value=DEFAULT_CALL_COUNT)
        self.proxy_info = tk.StringVar(value="系统代理: 检测中…")
        self.ip_info = tk.StringVar(value="公网 IP: 检测中…")

        self.edit_en = tk.IntVar(value=DEFAULT_CALL_COUNT)
        self.edit_ch = tk.IntVar(value=DEFAULT_CALL_COUNT)
        self.edit_provider = tk.StringVar(value="deepseek")
        self.edit_model = tk.StringVar(value="")

        self._scale_vars: Dict[str, dict] = {}
        self._models_cache: Dict[str, List[str]] = {
            p: list(DEFAULT_MODELS.get(p, [])) for p in PROVIDERS
        }
        self._runner: Optional[BatchRunner] = None
        self._worker: Optional[threading.Thread] = None
        self._ip_cache: dict = {}

        self._build_ui()
        self._load_persisted()
        self.refresh_scales()
        self.refresh_models(silent=True)
        self._prefetch_all_models()
        proxy_util.apply_proxy_to_env()
        self.refresh_network_info()

    def _build_ui(self) -> None:
        header = ttk.Frame(self, style="Header.TFrame", padding=(18, 14))
        header.pack(fill=tk.X)
        ttk.Label(header, text="量表 API 批量作答工具", style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            header,
            text="按量表指定模型 · 题目顺序随机 · 结果按原序保存 · 自动跟随系统代理",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(self, padding=(14, 12))
        body.pack(fill=tk.BOTH, expand=True)

        paths = make_card(body, "数据与结果路径")
        paths.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(paths, text="数据目录", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(paths, textvariable=self.data_dir).grid(
            row=0, column=1, sticky="we", padx=8, pady=3
        )
        ttk.Button(
            paths, text="浏览", command=self._browse_data, style="Ghost.TButton"
        ).grid(row=0, column=2, padx=2)
        ttk.Button(
            paths, text="刷新量表", command=self.refresh_scales, style="Accent.TButton"
        ).grid(row=0, column=3, padx=(6, 0))

        ttk.Label(paths, text="结果目录", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w"
        )
        ttk.Entry(paths, textvariable=self.results_dir).grid(
            row=1, column=1, sticky="we", padx=8, pady=3
        )
        ttk.Button(
            paths, text="浏览", command=self._browse_results, style="Ghost.TButton"
        ).grid(row=1, column=2, padx=2)
        paths.columnconfigure(1, weight=1)

        net = make_card(body, "网络状态")
        net.pack(fill=tk.X, pady=(0, 10))
        net_left = ttk.Frame(net, style="Surface.TFrame")
        net_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(net_left, textvariable=self.ip_info, style="Status.TLabel").pack(
            anchor="w", fill=tk.X, pady=(0, 6)
        )
        ttk.Label(
            net_left, textvariable=self.proxy_info, style="StatusProxy.TLabel"
        ).pack(anchor="w", fill=tk.X)
        ttk.Button(
            net,
            text="重新检测",
            command=self.refresh_network_info,
            style="Ghost.TButton",
        ).pack(side=tk.RIGHT, padx=(12, 0), anchor="n")

        api = make_card(body, "默认 API（可批量应用到各量表）")
        api.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(api, text="提供方", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", pady=3
        )
        providers = ttk.Combobox(
            api,
            textvariable=self.provider,
            values=list(PROVIDERS),
            state="readonly",
            width=14,
        )
        providers.grid(row=0, column=1, sticky="w", padx=(6, 14))
        providers.bind("<<ComboboxSelected>>", lambda _e: self.refresh_models())

        ttk.Label(api, text="模型", style="Muted.TLabel").grid(
            row=0, column=2, sticky="w"
        )
        self.model_box = ttk.Combobox(api, textvariable=self.model, width=34)
        self.model_box.grid(row=0, column=3, sticky="we", padx=6)
        ttk.Button(
            api, text="刷新模型", command=self.refresh_models, style="Ghost.TButton"
        ).grid(row=0, column=4, padx=4)
        ttk.Button(
            api, text="API 密钥", command=self._edit_api_keys, style="Accent.TButton"
        ).grid(row=0, column=5, padx=(4, 0))

        ttk.Label(api, text="温度", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=6
        )
        ttk.Spinbox(
            api,
            from_=0.0,
            to=2.0,
            increment=0.1,
            textvariable=self.temperature,
            width=8,
        ).grid(row=1, column=1, sticky="w", padx=(6, 14))
        ttk.Label(api, text="并发数", style="Muted.TLabel").grid(
            row=1, column=2, sticky="w"
        )
        ttk.Spinbox(api, from_=1, to=50, textvariable=self.concurrency, width=8).grid(
            row=1, column=3, sticky="w", padx=6
        )
        ttk.Label(
            api, text="同时进行的 API 调用数，默认 5", style="Hint.TLabel"
        ).grid(row=1, column=4, columnspan=2, sticky="w", padx=4)
        api.columnconfigure(3, weight=1)

        batch = make_card(body, "批量设置")
        batch.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(batch, text="英文次数", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Spinbox(
            batch, from_=0, to=100000, textvariable=self.batch_en, width=8
        ).grid(row=0, column=1, padx=(6, 16))
        ttk.Label(batch, text="中文次数", style="Muted.TLabel").grid(
            row=0, column=2, sticky="w"
        )
        ttk.Spinbox(
            batch, from_=0, to=100000, textvariable=self.batch_ch, width=8
        ).grid(row=0, column=3, padx=(6, 16))
        ttk.Button(
            batch,
            text="应用次数到全部",
            command=self._apply_batch_counts,
            style="Ghost.TButton",
        ).grid(row=0, column=4, padx=4)
        ttk.Button(
            batch,
            text="应用默认 API 到全部",
            command=self._apply_batch_api,
            style="Accent.TButton",
        ).grid(row=0, column=5, padx=4)

        scales_card = make_card(body, "量表列表（单击「启用」切换；选中后可单独配置）")
        scales_card.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        tree_wrap = ttk.Frame(scales_card, style="Surface.TFrame")
        tree_wrap.pack(fill=tk.BOTH, expand=True)

        cols = (
            "enabled",
            "name",
            "provider",
            "model",
            "en_n",
            "ch_n",
            "en_count",
            "ch_count",
        )
        self.tree = ttk.Treeview(
            tree_wrap,
            columns=cols,
            show="headings",
            height=7,
            selectmode="browse",
        )
        headings = {
            "enabled": "启用",
            "name": "量表名称",
            "provider": "API",
            "model": "模型",
            "en_n": "英文题数",
            "ch_n": "中文题数",
            "en_count": "英文次数",
            "ch_count": "中文次数",
        }
        widths = {
            "enabled": 50,
            "name": 210,
            "provider": 90,
            "model": 190,
            "en_n": 70,
            "ch_n": 70,
            "en_count": 80,
            "ch_count": 80,
        }
        for c in cols:
            self.tree.heading(c, text=headings[c])
            self.tree.column(c, width=widths[c], anchor="center")
        self.tree.column("name", anchor="w")
        self.tree.column("model", anchor="w")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.tree.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<Double-1>", self._edit_scale_row)
        self.tree.bind("<Button-1>", self._toggle_enabled)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        edit = ttk.Frame(scales_card, style="Surface.TFrame")
        edit.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(edit, text="选中量表配置", style="Muted.TLabel").pack(side=tk.LEFT)
        self.edit_provider_box = ttk.Combobox(
            edit,
            textvariable=self.edit_provider,
            values=list(PROVIDERS),
            state="readonly",
            width=12,
        )
        self.edit_provider_box.pack(side=tk.LEFT, padx=(8, 4))
        self.edit_provider_box.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_edit_provider_change()
        )
        self.edit_model_box = ttk.Combobox(edit, textvariable=self.edit_model, width=28)
        self.edit_model_box.pack(side=tk.LEFT, padx=4)
        ttk.Label(edit, text="英文", style="Muted.TLabel").pack(
            side=tk.LEFT, padx=(8, 2)
        )
        ttk.Spinbox(edit, from_=0, to=100000, textvariable=self.edit_en, width=7).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Label(edit, text="中文", style="Muted.TLabel").pack(
            side=tk.LEFT, padx=(8, 2)
        )
        ttk.Spinbox(edit, from_=0, to=100000, textvariable=self.edit_ch, width=7).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(
            edit,
            text="应用到选中",
            command=self._apply_selected,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=(10, 0))

        prog = ttk.Frame(body)
        prog.pack(fill=tk.X, pady=(2, 8))
        self.progress = ttk.Progressbar(prog, mode="determinate")
        self.progress.pack(fill=tk.X, expand=True, side=tk.LEFT)
        self.progress_label = ttk.Label(prog, text="0 / 0", style="Muted.TLabel")
        self.progress_label.pack(side=tk.RIGHT, padx=(10, 0))

        log_card = make_card(body, "运行日志（同步写入 logs/）")
        log_card.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        log_wrap = ttk.Frame(log_card, style="Surface.TFrame")
        log_wrap.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_wrap, height=8, wrap=tk.WORD)
        style_log_text(self.log_text)
        self.log_text.tag_configure("ok", foreground="#4ADE80")
        self.log_text.tag_configure("err", foreground="#FCA5A5")
        self.log_text.tag_configure("info", foreground="#7DD3FC")
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        log_sb = ttk.Scrollbar(log_wrap, command=self.log_text.yview)
        log_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_sb.set)

        footer = ttk.Frame(self, style="Footer.TFrame", padding=(14, 10))
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        self.start_btn = ttk.Button(
            footer, text="▶  开始运行", command=self.start_run, style="Primary.TButton"
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_btn = ttk.Button(
            footer,
            text="■  停止",
            command=self.stop_run,
            state=tk.DISABLED,
            style="Danger.TButton",
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            footer, text="保存配置", command=self._persist, style="Ghost.TButton"
        ).pack(side=tk.LEFT)
        ttk.Button(
            footer, text="打开日志目录", command=self._open_logs, style="Ghost.TButton"
        ).pack(side=tk.RIGHT)
        ttk.Button(
            footer,
            text="打开结果目录",
            command=self._open_results,
            style="Ghost.TButton",
        ).pack(side=tk.RIGHT, padx=(0, 8))


    def _browse_data(self) -> None:
        initial = resolve_path(self.data_dir.get())
        path = filedialog.askdirectory(initialdir=str(initial))
        if path:
            self.data_dir.set(to_rel_path(path))
            self.refresh_scales()

    def _browse_results(self) -> None:
        initial = resolve_path(self.results_dir.get())
        path = filedialog.askdirectory(initialdir=str(initial))
        if path:
            self.results_dir.set(to_rel_path(path))

    def refresh_network_info(self) -> None:
        self.ip_info.set("公网 IP: 检测中…")
        self.proxy_info.set("系统代理: 检测中…")
        self._log("正在检测公网 IP 与系统代理…")

        def work() -> None:
            proxy_util.apply_proxy_to_env()
            proxy = proxy_util.detect_system_proxy()
            try:
                info = detect_public_ip_info()
            except Exception as exc:
                info = {"ip": None, "location": f"查询失败: {exc}", "org": ""}
            self._ip_cache = info

            def apply() -> None:
                ip = info.get("ip") or "未知"
                loc = info.get("location") or "未知"
                org = info.get("org") or ""
                org_part = f"  |  运营商: {org}" if org else ""
                self.ip_info.set(f"公网 IP: {ip}  |  位置: {loc}{org_part}")
                if proxy:
                    self.proxy_info.set(f"系统代理: {proxy}")
                else:
                    self.proxy_info.set("系统代理: 未检测到（将直连）")
                self._log(network_status_text(info, proxy))

            self.after(0, apply)

        threading.Thread(target=work, daemon=True).start()



    def _edit_api_keys(self) -> None:
        win = tk.Toplevel(self)
        win.title("API 密钥设置")
        win.geometry("680x560")
        win.configure(bg="#EEF2F6")
        win.transient(self)
        win.grab_set()

        head = ttk.Frame(win, style="Header.TFrame", padding=(14, 10))
        head.pack(fill=tk.X)
        ttk.Label(head, text="API 密钥", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            head,
            text="可分别为 DeepSeek / OpenAI / Qwen / Gemini 填写 API Key 与 Base URL",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(win, padding=14)
        body.pack(fill=tk.BOTH, expand=True)

        fields: Dict[str, Dict[str, tk.StringVar]] = {}
        for provider in ("deepseek", "openai", "qwen", "gemini"):
            card = make_card(body, provider.upper())
            card.pack(fill=tk.X, pady=(0, 10))
            cred = get_api_credentials(provider)
            key_var = tk.StringVar(value=cred.get("api_key", ""))
            url_var = tk.StringVar(value=cred.get("base_url", ""))
            fields[provider] = {"api_key": key_var, "base_url": url_var}
            ttk.Label(card, text="API Key", style="Muted.TLabel").grid(
                row=0, column=0, sticky="w"
            )
            ttk.Entry(card, textvariable=key_var, width=70, show="*").grid(
                row=0, column=1, sticky="we", padx=8, pady=3
            )
            ttk.Label(card, text="Base URL", style="Muted.TLabel").grid(
                row=1, column=0, sticky="w"
            )
            ttk.Entry(card, textvariable=url_var, width=70).grid(
                row=1, column=1, sticky="we", padx=8, pady=3
            )
            card.columnconfigure(1, weight=1)

        def save() -> None:
            cfg = load_user_config()
            existing = dict(cfg.get("api_keys") or {})
            for p, v in fields.items():
                existing[p] = {
                    "api_key": v["api_key"].get().strip(),
                    "base_url": v["base_url"].get().strip(),
                }
            cfg["api_keys"] = existing
            save_user_config(cfg)
            self._log("API 密钥已保存到 user_config.json")
            win.destroy()
            self.refresh_models()
            self._prefetch_all_models()

        btns = ttk.Frame(body)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="保存", command=save, style="Primary.TButton").pack(
            side=tk.RIGHT
        )


    def _log(self, msg: str) -> None:
        tag = "info"
        if "[成功" in msg or "完成" in msg or "已加载" in msg:
            tag = "ok"
        elif "[失败" in msg or "[重试" in msg or "错误" in msg or "失败" in msg:
            tag = "err"
        self.log_text.insert(tk.END, msg + "\n", tag)
        self.log_text.see(tk.END)

    def _default_model_for(self, provider: str) -> str:
        models = self._models_cache.get(provider) or list(
            DEFAULT_MODELS.get(provider, [])
        )
        if self.provider.get() == provider and self.model.get().strip():
            return self.model.get().strip()
        return models[0] if models else ""

    def _refresh_tree_row(self, name: str) -> None:
        meta = self._scale_vars[name]
        sc = next((s for s in self.scales if s.name == name), None)
        self.tree.item(
            name,
            values=(
                "✓" if meta.get("enabled", True) else "",
                name,
                meta.get("provider", ""),
                meta.get("model", ""),
                sc.en.n_items if sc and sc.en else 0,
                sc.ch.n_items if sc and sc.ch else 0,
                meta.get("en_count", 0),
                meta.get("ch_count", 0),
            ),
        )

    def refresh_scales(self) -> None:
        data_path = resolve_path(self.data_dir.get())
        self.scales = discover_scales(data_path)
        for item in self.tree.get_children():
            self.tree.delete(item)
        old = dict(self._scale_vars)
        self._scale_vars.clear()
        for sc in self.scales:
            prev = old.get(sc.name, {})
            provider = prev.get("provider") or self.provider.get() or "deepseek"
            model = prev.get("model") or self._default_model_for(provider)
            self._scale_vars[sc.name] = {
                "enabled": prev.get("enabled", True),
                "en_count": prev.get("en_count", self.batch_en.get()),
                "ch_count": prev.get("ch_count", self.batch_ch.get()),
                "en_provider": prev.get("en_provider") or provider,
                "en_model": prev.get("en_model") or model,
                "ch_provider": prev.get("ch_provider") or provider,
                "ch_model": prev.get("ch_model") or model,
                "provider": provider,
                "model": model,
            }
            self.tree.insert("", tk.END, iid=sc.name, values=())
            self._refresh_tree_row(sc.name)
        self._log(f"已加载 {len(self.scales)} 个量表（目录: {data_path}）")

    def _set_models_for_provider(
        self, provider: str, models: List[str], *, update_global: bool = False
    ) -> None:
        self._models_cache[provider] = models
        if update_global and self.provider.get() == provider:
            self.model_box["values"] = models
            if models and self.model.get() not in models:
                self.model.set(models[0])
            elif not self.model.get() and models:
                self.model.set(models[0])
        if self.edit_provider.get() == provider:
            self.edit_model_box["values"] = models
            if models and self.edit_model.get() not in models:
                self.edit_model.set(models[0])

    def refresh_models(self, silent: bool = False) -> None:
        provider = self.provider.get()
        if not silent:
            self._log(f"正在查询 {provider} 模型列表…")

        def work() -> None:
            try:
                cfg = get_api_credentials(provider)
                models = list_models(provider, cfg.get("api_key"), cfg.get("base_url"))
                if not models:
                    models = list(DEFAULT_MODELS.get(provider, []))
            except Exception as exc:
                models = list(DEFAULT_MODELS.get(provider, []))
                self.after(
                    0, lambda: self._log(f"模型列表查询失败，使用默认列表: {exc}")
                )

            def apply() -> None:
                self._set_models_for_provider(provider, models, update_global=True)
                if not silent:
                    self._log(f"已加载 {provider} 的 {len(models)} 个模型")

            self.after(0, apply)

        threading.Thread(target=work, daemon=True).start()

    def _prefetch_all_models(self) -> None:
        def work() -> None:
            for provider in PROVIDERS:
                try:
                    cfg = get_api_credentials(provider)
                    models = list_models(
                        provider, cfg.get("api_key"), cfg.get("base_url")
                    )
                    if not models:
                        models = list(DEFAULT_MODELS.get(provider, []))
                except Exception:
                    models = list(DEFAULT_MODELS.get(provider, []))
                self.after(
                    0,
                    lambda p=provider, m=models: self._set_models_for_provider(p, m),
                )

        threading.Thread(target=work, daemon=True).start()

    def _on_edit_provider_change(self) -> None:
        provider = self.edit_provider.get()
        models = self._models_cache.get(provider) or list(
            DEFAULT_MODELS.get(provider, [])
        )
        self.edit_model_box["values"] = models
        if models:
            self.edit_model.set(models[0])
        else:
            self.edit_model.set("")

    def _on_tree_select(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        self._load_edit_from_scale(sel[0])

    def _load_edit_from_scale(self, name: str) -> None:
        meta = self._scale_vars.get(name, {})
        provider = meta.get("provider") or self.provider.get()
        model = meta.get("model") or ""
        self.edit_provider.set(provider)
        models = self._models_cache.get(provider) or list(
            DEFAULT_MODELS.get(provider, [])
        )
        self.edit_model_box["values"] = models
        self.edit_model.set(model if model else (models[0] if models else ""))
        self.edit_en.set(meta.get("en_count", DEFAULT_CALL_COUNT))
        self.edit_ch.set(meta.get("ch_count", DEFAULT_CALL_COUNT))

    def _apply_batch_counts(self) -> None:
        for name, meta in self._scale_vars.items():
            meta["en_count"] = int(self.batch_en.get())
            meta["ch_count"] = int(self.batch_ch.get())
            self._refresh_tree_row(name)
        self._log(
            f"已批量设置次数：英文={self.batch_en.get()}，中文={self.batch_ch.get()}"
        )

    def _apply_batch_api(self) -> None:
        provider = self.provider.get()
        model = self.model.get().strip()
        if not model:
            messagebox.showerror("错误", "请先选择默认模型")
            return
        for name, meta in self._scale_vars.items():
            meta["provider"] = provider
            meta["model"] = model
            meta["en_provider"] = provider
            meta["en_model"] = model
            meta["ch_provider"] = provider
            meta["ch_model"] = model
            self._refresh_tree_row(name)
        self._log(f"已将默认 API 应用到全部量表（中英文）：{provider} / {model}")

    def _apply_selected(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("提示", "请先选中一个量表")
            return
        name = sel[0]
        provider = self.edit_provider.get().strip()
        model = self.edit_model.get().strip()
        if not provider or not model:
            messagebox.showerror("错误", "请为选中量表指定 API 提供方和模型")
            return
        meta = self._scale_vars[name]
        meta["provider"] = provider
        meta["model"] = model
        meta["en_count"] = int(self.edit_en.get())
        meta["ch_count"] = int(self.edit_ch.get())
        self._refresh_tree_row(name)
        self._log(
            f"已更新 {name}: {provider}/{model}, "
            f"en={meta['en_count']}, ch={meta['ch_count']}"
        )

    def _toggle_enabled(self, event) -> None:
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        col = self.tree.identify_column(event.x)
        if col != "#1":
            return
        row = self.tree.identify_row(event.y)
        if not row:
            return
        meta = self._scale_vars[row]
        meta["enabled"] = not meta["enabled"]
        self._refresh_tree_row(row)

    def _edit_scale_row(self, _event) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        self._load_edit_from_scale(sel[0])


    def _on_progress(self, key: str, done: int, total: int) -> None:
        if key == "__all__":
            self.progress["maximum"] = max(total, 1)
            self.progress["value"] = done
            self.progress_label.config(text=f"{done} / {total}")

    def start_run(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showwarning("提示", "任务正在运行中")
            return
        if not self.scales:
            messagebox.showerror("错误", "未找到量表，请检查数据目录")
            return

        missing = []
        for name, meta in self._scale_vars.items():
            if not meta.get("enabled"):
                continue
            en_p = meta.get("en_provider") or meta.get("provider")
            en_m = meta.get("en_model") or meta.get("model")
            ch_p = meta.get("ch_provider") or meta.get("provider")
            ch_m = meta.get("ch_model") or meta.get("model")
            if (int(meta.get("en_count", 0)) > 0 and (not en_p or not str(en_m or "").strip())) or (
                int(meta.get("ch_count", 0)) > 0 and (not ch_p or not str(ch_m or "").strip())
            ):
                missing.append(name)
        if missing:
            messagebox.showerror(
                "错误",
                "以下启用的量表未完整指定中/英文 API/模型：\n" + "\n".join(missing),
            )
            return

        try:
            concurrency = int(self.concurrency.get())
        except (TypeError, ValueError, tk.TclError):
            concurrency = DEFAULT_CONCURRENCY
        if concurrency < 1:
            messagebox.showerror("错误", "并发数至少为 1")
            return

        proxy_util.apply_proxy_to_env()
        self.refresh_network_info()

        scale_cfgs: Dict[str, ScaleRunConfig] = {}
        for name, meta in self._scale_vars.items():
            legacy_p = str(meta.get("provider") or self.provider.get())
            legacy_m = str(meta.get("model") or self.model.get()).strip()
            scale_cfgs[name] = ScaleRunConfig(
                scale_name=name,
                en_count=int(meta["en_count"]),
                ch_count=int(meta["ch_count"]),
                enabled=bool(meta["enabled"]),
                en_provider=str(meta.get("en_provider") or legacy_p),
                en_model=str(meta.get("en_model") or legacy_m).strip(),
                ch_provider=str(meta.get("ch_provider") or legacy_p),
                ch_model=str(meta.get("ch_model") or legacy_m).strip(),
                provider=legacy_p,
                model=legacy_m,
            )

        config = RunnerConfig(
            temperature=float(self.temperature.get()),
            results_root=resolve_path(self.results_dir.get()),
            logs_root=LOGS_DIR,
            concurrency=concurrency,
            default_provider=self.provider.get(),
            default_model=self.model.get().strip(),
            scale_cfgs=scale_cfgs,
        )
        self._runner = BatchRunner(
            scales=self.scales,
            config=config,
            log=lambda m: self.after(0, self._log, m),
            progress=lambda k, d, t: self.after(0, self._on_progress, k, d, t),
        )
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress["value"] = 0
        self.progress_label.config(text="0/0")
        self._persist()

        def work() -> None:
            try:
                assert self._runner is not None
                self._runner.run()
            finally:
                self.after(0, self._run_finished)

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()
        self._log(f"任务已启动（并发={concurrency}）…")

    def stop_run(self) -> None:
        if self._runner:
            self._runner.request_stop()
            self._log("正在请求停止…")

    def _run_finished(self) -> None:
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._log("运行结束。")

    def _open_path(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        import os
        import platform
        import subprocess

        system = platform.system()
        if system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def _open_results(self) -> None:
        self._open_path(resolve_path(self.results_dir.get()))

    def _open_logs(self) -> None:
        self._open_path(LOGS_DIR)

    def _persist(self) -> None:
        cfg = load_user_config()
        cfg.update(
            {
                "data_dir": to_rel_path(self.data_dir.get()),
                "results_dir": to_rel_path(self.results_dir.get()),
                "provider": self.provider.get(),
                "model": self.model.get(),
                "temperature": float(self.temperature.get()),
                "concurrency": int(self.concurrency.get()),
                "batch_en": int(self.batch_en.get()),
                "batch_ch": int(self.batch_ch.get()),
                "scale_vars": self._scale_vars,
            }
        )
        save_user_config(cfg)
        self.data_dir.set(cfg["data_dir"])
        self.results_dir.set(cfg["results_dir"])
        self._log("配置已保存到 user_config.json")

    def _load_persisted(self) -> None:
        cfg = load_user_config()
        if not cfg:
            return
        data_dir = to_rel_path(cfg.get("data_dir", DEFAULT_DATA_DIR)) or DEFAULT_DATA_DIR
        results_dir = to_rel_path(cfg.get("results_dir", DEFAULT_RESULTS_DIR)) or DEFAULT_RESULTS_DIR
        self.data_dir.set(data_dir)
        self.results_dir.set(results_dir)
        self.provider.set(cfg.get("provider", self.provider.get()))
        self.model.set(cfg.get("model", ""))
        self.temperature.set(cfg.get("temperature", DEFAULT_TEMPERATURE))
        self.concurrency.set(cfg.get("concurrency", DEFAULT_CONCURRENCY))
        self.batch_en.set(cfg.get("batch_en", DEFAULT_CALL_COUNT))
        self.batch_ch.set(cfg.get("batch_ch", DEFAULT_CALL_COUNT))
        self._scale_vars = cfg.get("scale_vars", {})


def run_app() -> None:
    app = ScaleApp()
    app.mainloop()
