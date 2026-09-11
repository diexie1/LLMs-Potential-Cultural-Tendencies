"""批量调用、失败重试、按原序保存结果（支持并发、按量表指定 API）。"""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import proxy_util
from .api_client import call_api
from .config import (
    DEFAULT_CALL_COUNT,
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY_SEC,
    LOGS_DIR,
    RESULTS_DIR,
    SYSTEM_PROMPT,
    display_path,
    get_api_credentials,
    sanitize_name,
)
from .file_logger import RunFileLogger
from .prompting import (
    build_prompt,
    build_result_payload,
    parse_scores,
    remap_to_original,
    shuffle_order,
)
from .scale_loader import ScaleFile, ScaleSheet


LogFn = Callable[[str], None]
ProgressFn = Callable[[str, int, int], None]


@dataclass
class ScaleRunConfig:
    scale_name: str
    en_count: int = DEFAULT_CALL_COUNT
    ch_count: int = DEFAULT_CALL_COUNT
    enabled: bool = True
    en_provider: str = "deepseek"
    en_model: str = "deepseek-v4-pro"
    ch_provider: str = "deepseek"
    ch_model: str = "deepseek-v4-pro"
    # 兼容旧配置：未分语言时回退用这两项
    provider: str = ""
    model: str = ""


@dataclass
class RunnerConfig:
    temperature: float = 0.7
    results_root: Path = field(default_factory=lambda: RESULTS_DIR)
    logs_root: Path = field(default_factory=lambda: LOGS_DIR)
    concurrency: int = DEFAULT_CONCURRENCY
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY_SEC
    # 全局默认，仅在量表未单独指定时使用
    default_provider: str = "deepseek"
    default_model: str = "deepseek-v4-pro"
    scale_cfgs: Dict[str, ScaleRunConfig] = field(default_factory=dict)


class BatchRunner:
    def __init__(
        self,
        scales: List[ScaleFile],
        config: RunnerConfig,
        log: Optional[LogFn] = None,
        progress: Optional[ProgressFn] = None,
    ):
        self.scales = scales
        self.config = config
        self.log = log or (lambda m: print(m))
        self.progress = progress or (lambda *_: None)
        self._stop = threading.Event()
        self._file_logger: Optional[RunFileLogger] = None

    def request_stop(self) -> None:
        self._stop.set()

    def _ui_log(self, message: str) -> None:
        self.log(message)
        if self._file_logger:
            self._file_logger.log(message)

    def _scale_api(self, scale_name: str, language: str) -> Tuple[str, str]:
        cfg = self.config.scale_cfgs.get(scale_name)
        legacy_p = self.config.default_provider
        legacy_m = self.config.default_model
        if cfg:
            legacy_p = cfg.provider or cfg.en_provider or legacy_p
            legacy_m = cfg.model or cfg.en_model or legacy_m
            if language == "ch":
                provider = cfg.ch_provider or cfg.provider or legacy_p
                model = cfg.ch_model or cfg.model or legacy_m
            else:
                provider = cfg.en_provider or cfg.provider or legacy_p
                model = cfg.en_model or cfg.model or legacy_m
        else:
            provider, model = legacy_p, legacy_m
        return (provider or "").strip(), (model or "").strip()

    def _result_path(
        self,
        scale_name: str,
        language: str,
        order_id: int,
        model: str,
    ) -> Path:
        lang_dir = "ch" if language == "ch" else "en"
        model_safe = sanitize_name(model)
        scale_safe = sanitize_name(scale_name)
        temp_str = str(self.config.temperature).replace(".", "p")
        folder = (
            Path(self.config.results_root)
            / scale_safe
            / lang_dir
            / model_safe
        )
        folder.mkdir(parents=True, exist_ok=True)
        fname = f"order_{order_id:04d}_temperature_{temp_str}.json"
        return folder / fname

    def _one_call(
        self,
        scale: ScaleFile,
        sheet: ScaleSheet,
        language: str,
        order_id: int,
        provider: str,
        model: str,
    ) -> Path:
        seed = random.randint(0, 2**31 - 1)
        rng = random.Random(seed)
        order = shuffle_order(sheet.n_items, rng)
        prompt = build_prompt(sheet, order)
        cred = get_api_credentials(provider)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            if self._stop.is_set():
                raise RuntimeError("用户已停止")
            try:
                proxy_util.apply_proxy_to_env()
                if self._file_logger:
                    self._file_logger.log(
                        f"开始请求 {scale.name}/{language} #{order_id} "
                        f"provider={provider} model={model} "
                        f"attempt={attempt} seed={seed}"
                    )
                raw = call_api(
                    provider,
                    prompt,
                    model=model,
                    temperature=self.config.temperature,
                    api_key=cred.get("api_key"),
                    base_url=cred.get("base_url"),
                    system_prompt=SYSTEM_PROMPT,
                )
                shuffled_scores = parse_scores(raw, sheet.n_items)
                scores = remap_to_original(shuffled_scores, order)
                payload = build_result_payload(
                    scale_name=scale.name,
                    language=language,
                    provider=provider,
                    model=model,
                    temperature=self.config.temperature,
                    order_id=order_id,
                    seed=seed,
                    order=order,
                    sheet=sheet,
                    scores_original=scores,
                    raw_response=raw,
                    prompt=prompt,
                )
                path = self._result_path(scale.name, language, order_id, model)
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                csv_path = path.with_suffix(".csv")
                self._write_csv(csv_path, payload)
                if self._file_logger:
                    self._file_logger.log_call(
                        scale_name=scale.name,
                        language=language,
                        order_id=order_id,
                        attempt=attempt,
                        prompt=prompt,
                        response=raw,
                        scores=list(scores),
                        result_path=display_path(path),
                        seed=seed,
                        shuffle_order=[i + 1 for i in order],
                    )
                return path
            except Exception as exc:
                last_err = exc
                msg = (
                    f"[重试 {attempt}/{self.config.max_retries}] "
                    f"{scale.name}/{language} #{order_id} "
                    f"[{provider}/{model}]: {exc}"
                )
                self._ui_log(msg)
                if self._file_logger:
                    self._file_logger.log_call(
                        scale_name=scale.name,
                        language=language,
                        order_id=order_id,
                        attempt=attempt,
                        prompt=prompt,
                        error=str(exc),
                        seed=seed,
                        shuffle_order=[i + 1 for i in order],
                    )
                time.sleep(self.config.retry_delay * attempt)
        raise RuntimeError(
            f"调用失败已达最大重试次数: {scale.name}/{language} #{order_id} "
            f"[{provider}/{model}]: {last_err}"
        )

    @staticmethod
    def _write_csv(path: Path, payload: dict) -> None:
        lines = ["index,question,score"]
        for item in payload["items"]:
            q = str(item["question"]).replace('"', '""')
            score = "" if item["score"] is None else item["score"]
            lines.append(f'{item["index"]},"{q}",{score}')
        path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    def _run_group(
        self,
        scale: ScaleFile,
        sheet: ScaleSheet,
        language: str,
        count: int,
        provider: str,
        model: str,
        grand_done: List[int],
        grand_total: int,
    ) -> None:
        task_key = f"{scale.name}/{language}"
        self._ui_log(
            f"—— 开始 {task_key} | API={provider} | 模型={model}，"
            f"目标成功 {count} 次，并发={self.config.concurrency} ——"
        )

        success = 0
        order_id = 1
        while self._result_path(scale.name, language, order_id, model).exists():
            order_id += 1

        id_lock = threading.Lock()
        next_order_id = order_id
        state_lock = threading.Lock()

        def submit_job(executor: ThreadPoolExecutor):
            nonlocal next_order_id
            with id_lock:
                oid = next_order_id
                next_order_id += 1

            def job() -> Tuple[bool, Optional[Path], Optional[str], int]:
                try:
                    path = self._one_call(
                        scale, sheet, language, oid, provider, model
                    )
                    return True, path, None, oid
                except Exception as exc:
                    return False, None, str(exc), oid

            return executor.submit(job)

        workers = max(1, int(self.config.concurrency))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = set()
            while True:
                if self._stop.is_set():
                    self._ui_log("已停止。")
                    for fut in pending:
                        fut.cancel()
                    return

                with state_lock:
                    cur_success = success

                while (
                    len(pending) < workers
                    and cur_success + len(pending) < count
                    and not self._stop.is_set()
                ):
                    pending.add(submit_job(executor))

                if not pending:
                    break

                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    ok, path, err, oid = fut.result()
                    if ok and path is not None:
                        with state_lock:
                            if success < count:
                                success += 1
                                grand_done[0] += 1
                                s = success
                                g = grand_done[0]
                            else:
                                s = success
                                g = grand_done[0]
                        self._ui_log(
                            f"[成功 {s}/{count}] {task_key} [{provider}/{model}] "
                            f"#{oid} -> {path.name}"
                        )
                        self.progress(task_key, s, count)
                        self.progress("__all__", g, grand_total)
                    else:
                        if err and "用户已停止" in err:
                            self._ui_log("已停止。")
                            return
                        self._ui_log(
                            f"[失败] {task_key} #{oid}: {err}，将继续补齐成功次数…"
                        )

                with state_lock:
                    if success >= count:
                        break

        self._ui_log(f"—— 完成 {task_key}：成功 {success}/{count} ——")

    def run(self) -> None:
        logs_root = Path(self.config.logs_root)
        logs_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_root / f"run_{stamp}.log"
        self._file_logger = RunFileLogger(log_path)
        self._ui_log(f"详细日志文件: {display_path(log_path)}")

        proxy = proxy_util.apply_proxy_to_env()
        self._ui_log(
            proxy_util.proxy_status_text() if proxy else "未使用代理，直连 API"
        )
        self._ui_log(
            f"温度={self.config.temperature}, 并发={self.config.concurrency}"
        )

        tasks = []
        for scale in self.scales:
            cfg = self.config.scale_cfgs.get(scale.name)
            if cfg is None:
                cfg = ScaleRunConfig(
                    scale_name=scale.name,
                    en_provider=self.config.default_provider,
                    en_model=self.config.default_model,
                    ch_provider=self.config.default_provider,
                    ch_model=self.config.default_model,
                )
            if not cfg.enabled:
                continue
            if scale.en and cfg.en_count > 0:
                provider, model = self._scale_api(scale.name, "en")
                if not provider or not model:
                    self._ui_log(f"[跳过] {scale.name}/en: 未指定 API 提供方或模型")
                else:
                    tasks.append((scale, scale.en, "en", cfg.en_count, provider, model))
            if scale.ch and cfg.ch_count > 0:
                provider, model = self._scale_api(scale.name, "ch")
                if not provider or not model:
                    self._ui_log(f"[跳过] {scale.name}/ch: 未指定 API 提供方或模型")
                else:
                    tasks.append((scale, scale.ch, "ch", cfg.ch_count, provider, model))

        if not tasks:
            self._ui_log("没有待执行的任务，请检查量表勾选、调用次数与 API 设置。")
            self._file_logger.close_note("无任务，结束")
            return

        for scale, _sheet, language, count, provider, model in tasks:
            self._ui_log(
                f"计划任务: {scale.name}/{language} x{count} -> {provider}/{model}"
            )

        grand_total = sum(t[3] for t in tasks)
        grand_done = [0]
        self._ui_log(f"开始执行，共 {grand_total} 次成功调用目标。")

        for scale, sheet, language, count, provider, model in tasks:
            if self._stop.is_set():
                break
            self._run_group(
                scale,
                sheet,
                language,
                count,
                provider,
                model,
                grand_done,
                grand_total,
            )
            if self._stop.is_set():
                break

        if self._stop.is_set():
            self._ui_log("运行被用户停止。")
            self._file_logger.close_note("用户停止")
        else:
            self._ui_log("全部任务完成。")
            self._file_logger.close_note("全部完成")
