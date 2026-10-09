"""批量调用、失败重试、按原序保存结果（支持并发、按量表指定 API）。"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import proxy_util
from . import __version__ as APP_VERSION
from .api_client import SHANGHAI_TZ, _safe_endpoint, call_api, fetch_model_catalog
from .condition_design import (
    build_condition_descriptor,
    condition_config_hash,
    normalize_condition_mode,
)
from .generation_config import (
    GenerationConfig,
    build_chat_request,
    simple_platform_generation_config,
    validate_generation_config,
)
from .model_profiles import build_model_profile, model_profile_key
from .provenance import (
    build_account_snapshot,
    build_network_snapshot,
    build_run_context,
    build_software_snapshot,
    canonical_hash,
    compare_network_snapshots,
    extract_error_metadata,
    file_sha256,
    sanitize_proxy,
)
from .config import (
    DEFAULT_CALL_COUNT,
    DEFAULT_CONCURRENCY,
    DEFAULT_SCALE_CONCURRENCY,
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
    DEFAULT_PROMPT_CONTRACT,
    build_trial_prompt,
    build_result_payload,
    classify_model_response,
    is_free_prompt_contract,
    normalize_prompt_contract,
    normalize_cultural_identity,
    parse_free_answers,
    parse_kohlberg_free_answers,
    remap_to_original,
)
from .scale_loader import ScaleFile, ScaleSheet, paired_shuffle_compatible
from .trial_results import export_trials
from .scale_profiles import (
    kohlberg_standalone_form,
    normalize_kohlberg_form,
    select_kohlberg_form,
)


LogFn = Callable[[str], None]
ProgressFn = Callable[[str, int, int], None]


def normalize_sample_mode(value: Any) -> str:
    """Return the only supported trial policy.

    ``valid`` was an older setting that created replacement trials until a
    requested number of parseable records had been collected.  It remains a
    migration alias so old configuration files can still start, but it is
    deliberately normalized to the fixed-plan policy and never replenishes
    trials.
    """

    raw = str(value or "planned").strip().lower()
    if raw in {"planned", "valid"}:
        return "planned"
    raise ValueError("试次策略已固定为 planned：不补齐、不替换试次")


class StorageError(RuntimeError):
    """The model response exists but could not be persisted locally."""


class AttemptBudgetExhausted(RuntimeError):
    """No group-level API attempt budget remains for a new request."""


def default_shuffle_items(scale_name: str) -> bool:
    """Shuffle by default; scale structure controls whether shuffling is allowed."""

    return True


def _timestamp_pair(moment: datetime) -> Tuple[str, str]:
    """Return one instant in unambiguous UTC and Asia/Shanghai forms."""

    utc = moment.astimezone(timezone.utc)
    return utc.isoformat(), utc.astimezone(SHANGHAI_TZ).isoformat()


def _parse_aware_timestamp(value: Any) -> Optional[datetime]:
    """Parse an offset-aware ISO timestamp, accepting a trailing ``Z``."""

    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    # A naive timestamp has no defensible timezone interpretation. Let the
    # caller use its request-completion fallback instead of guessing.
    return parsed if parsed.tzinfo is not None else None


def _normalize_recorded_timestamp(
    model_metadata: Dict[str, Any], fallback: datetime
) -> Tuple[str, str]:
    """Normalize provider/legacy recorded timestamps to both timezones."""

    candidate = model_metadata.get("recorded_at_utc")
    if candidate is None:
        candidate = model_metadata.get("recorded_at_local")
    recorded_at = _parse_aware_timestamp(candidate) or fallback
    return _timestamp_pair(recorded_at)


def _manifest_hash(manifest: Dict[str, Any]) -> str:
    """Hash manifest content without recursively hashing its own digest."""

    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    # None uses the profile default; the workbench derives this from scale rules.
    shuffle_items: Optional[bool] = None
    # Kohlberg MJI: all three forms by default, or one selected parallel form.
    mji_form: str = "all"


@dataclass
class RunnerConfig:
    # None means temperature is omitted and the provider/model chooses its
    # documented or runtime default (the natural condition).
    temperature: Optional[float] = 0.7
    results_root: Path = field(default_factory=lambda: RESULTS_DIR)
    logs_root: Path = field(default_factory=lambda: LOGS_DIR)
    concurrency: int = DEFAULT_CONCURRENCY
    scale_concurrency: int = DEFAULT_SCALE_CONCURRENCY
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY_SEC
    # 全局默认，仅在量表未单独指定时使用
    default_provider: str = "deepseek"
    default_model: str = "deepseek-v4-pro"
    # The preset is descriptive provenance only; generation_config remains the
    # authoritative request condition and is always stored in full.
    preset_id: str = "custom"
    # Legacy/direct callers are custom unless the Web workbench or a formal
    # preset supplies natural/controlled explicitly.
    condition_mode: str = "custom"
    # fixed | paired | per_scale (legacy direct-caller behavior)
    order_strategy: str = "per_scale"
    scale_cfgs: Dict[str, ScaleRunConfig] = field(default_factory=dict)
    generation_config: Optional[GenerationConfig] = None
    strict_capabilities: bool = False
    # 0 means count * max(1, max_retries), providing a bounded API-attempt
    # budget for the fixed trial plan.
    max_group_attempts: int = 0
    # Kept as a backward-compatible configuration field, but there is only one
    # policy now: submit the planned trial IDs once and preserve their status.
    sample_mode: str = "planned"
    # Master seed for deterministic item-order seeds.  None derives one from
    # run_id, so every run still has an auditable seed without global RNG use.
    random_seed: Optional[int] = None
    run_id: str = ""
    # The web workbench sends scale wording without generated answer constraints.
    prompt_contract: str = DEFAULT_PROMPT_CONTRACT
    cultural_identity: str = "none"
    # Provenance capture is enabled by the web workbench. Direct legacy
    # callers may disable network/catalog probes for offline test runs.
    capture_network: bool = True
    capture_model_catalog: bool = True
    network_guard: str = "warn"  # warn | stop

    def __post_init__(self) -> None:
        if int(self.concurrency) < 1:
            raise ValueError("concurrency 必须至少为 1")
        if int(self.scale_concurrency) < 1:
            raise ValueError("scale_concurrency 必须至少为 1")
        if int(self.max_retries) < 1:
            raise ValueError("max_retries 必须至少为 1")
        if float(self.retry_delay) < 0:
            raise ValueError("retry_delay 不能为负数")
        if int(self.max_group_attempts) < 0:
            raise ValueError("max_group_attempts 不能为负数")
        self.sample_mode = normalize_sample_mode(self.sample_mode)
        if self.network_guard not in {"warn", "stop"}:
            raise ValueError("network_guard 必须是 warn 或 stop")
        if self.order_strategy not in {"fixed", "paired", "per_scale"}:
            raise ValueError("order_strategy 必须是 fixed、paired 或 per_scale")
        self.condition_mode = normalize_condition_mode(self.condition_mode)
        self.prompt_contract = normalize_prompt_contract(self.prompt_contract)
        self.cultural_identity = normalize_cultural_identity(self.cultural_identity)


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
        self._progress_lock = threading.Lock()
        self._file_logger: Optional[RunFileLogger] = None
        self._resolved_proxy: Optional[str] = None
        self._run_context: Dict[str, Any] = {}
        self._manifest_path: Optional[Path] = None
        self._network_end: Optional[Dict[str, Any]] = None
        self._model_catalog_snapshot_ids: Dict[str, str] = {}
        self._account_snapshot_ids: Dict[str, str] = {}

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

    def _effective_sheet(self, scale: ScaleFile, sheet: ScaleSheet) -> ScaleSheet:
        """Apply scale-level structural settings before planning or sending.

        Kohlberg's ``mji_form`` changes the number and identity of response
        units, so it must be resolved before the manifest, trial plan, hash,
        and prompt are built. Other profiles pass through unchanged.
        """

        if sheet.profile_id != "kohlberg_mji_v1":
            return sheet
        if kohlberg_standalone_form(sheet):
            return sheet
        cfg = self.config.scale_cfgs.get(scale.name)
        selected = normalize_kohlberg_form(cfg.mji_form if cfg else "all")
        if selected == "all":
            return sheet
        # Avoid applying the same view twice when a direct caller passes the
        # already-selected sheet into ``_one_call``.
        if sheet.profile_version.endswith(f"-form-{selected}"):
            return sheet
        return select_kohlberg_form(sheet, selected)

    def _should_shuffle(
        self, scale_name: str, sheet: Optional[ScaleSheet] = None
    ) -> bool:
        """Resolve a shuffle setting while respecting profile safety rules."""

        if self.config.order_strategy == "fixed":
            return False
        if sheet is not None and not sheet.shuffle_supported:
            return False
        if self.config.order_strategy == "paired":
            scale = next((item for item in self.scales if item.name == scale_name), None)
            if scale is not None and not paired_shuffle_compatible(scale.en, scale.ch):
                return False
        cfg = self.config.scale_cfgs.get(scale_name)
        if cfg is not None and cfg.shuffle_items is not None:
            return bool(cfg.shuffle_items)
        if self.config.order_strategy == "paired":
            return True
        if sheet is not None:
            return bool(sheet.shuffle_supported)
        return default_shuffle_items(scale_name)

    def _result_path(
        self,
        scale_name: str,
        language: str,
        order_id: int,
        model: str,
        *,
        create_parent: bool = False,
    ) -> Path:
        lang_dir = "ch" if language == "ch" else "en"
        model_safe = sanitize_name(model)
        scale_safe = sanitize_name(scale_name)
        effective_temperature = self._effective_generation_config().temperature
        temp_str = (
            "provider_default"
            if effective_temperature is None
            else str(effective_temperature).replace(".", "p")
        )
        folder = Path(self.config.results_root)
        if self.config.run_id:
            folder = folder / sanitize_name(self.config.run_id)
        folder = folder / scale_safe / lang_dir / model_safe
        if create_parent:
            folder.mkdir(parents=True, exist_ok=True)
        fname = f"order_{order_id:04d}_temperature_{temp_str}.json"
        return folder / fname

    def _effective_generation_config(self) -> GenerationConfig:
        source = self.config.generation_config
        if source is None:
            source = {"temperature": self.config.temperature, "top_p": 1.0}
        return simple_platform_generation_config(
            source,
            fallback_temperature=(
                self.config.temperature
                if self.config.temperature is not None
                else 0.7
            ),
        )

    def _trial_seed(
        self, scale_name: str, language: str, order_id: int, shuffle_enabled: bool
    ) -> Optional[int]:
        """Derive a reproducible order seed, paired across languages if requested."""

        if not shuffle_enabled:
            return None
        master = str(self._effective_random_seed())
        if self.config.order_strategy == "paired":
            material = f"{master}|{scale_name}|{int(order_id)}|paired-trial-order-v1"
        else:
            material = f"{master}|{scale_name}|{language}|{int(order_id)}|trial-order-v1"
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**31 - 1)

    def _effective_random_seed(self) -> int:
        if self.config.random_seed is not None:
            return int(self.config.random_seed)
        material = self.config.run_id or "adhoc"
        return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)

    def _effective_order_seed(self) -> Optional[int]:
        """Return the seed that can affect this run's displayed item order."""

        if self.config.order_strategy == "fixed":
            return None
        return self._effective_random_seed()

    def _plan_path(self) -> Path:
        return (
            Path(self.config.results_root)
            / sanitize_name(self.config.run_id or "adhoc")
            / "plan.jsonl"
        )

    def _write_trial_plan(
        self,
        *,
        scale: ScaleFile,
        sheet: ScaleSheet,
        language: str,
        provider: str,
        model: str,
        order_ids: List[int],
    ) -> None:
        """Write the complete group's trial plan before submitting requests."""

        path = self._plan_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        shuffle_enabled = self._should_shuffle(scale.name, sheet)
        generation = self._effective_generation_config()
        condition_hash = self._condition_hash(
            scale, sheet, language, provider, model, generation
        )
        now_utc = datetime.now(timezone.utc).isoformat()
        with self._progress_lock, path.open("a", encoding="utf-8") as handle:
            for order_id in order_ids:
                trial_seed = self._trial_seed(
                    scale.name, language, order_id, shuffle_enabled
                )
                record = {
                    "plan_schema_version": "trial-plan-v2",
                    "run_id": self.config.run_id or None,
                    "trial_id": f"{scale.name}/{language}/{order_id:04d}",
                    "trial_key": f"{scale.name}/{language}/{order_id:04d}",
                    "pair_id": order_id if self.config.order_strategy == "paired" else None,
                    "paired_trial_key": (
                        f"{scale.name}/{order_id:04d}"
                        if self.config.order_strategy == "paired"
                        else None
                    ),
                    "scale_name": scale.name,
                    "language": language,
                    "order_id": order_id,
                    "order_strategy": self.config.order_strategy,
                    "provider": provider,
                    "requested_model": model,
                    "condition_mode": self.config.condition_mode,
                    "preset_id": self.config.preset_id,
                    "generation_config": generation.to_dict(),
                    "condition_descriptor": build_condition_descriptor(
                        self.config.condition_mode,
                        generation,
                        prompt_contract=self.config.prompt_contract,
                        cultural_identity=self.config.cultural_identity,
                    ),
                    "condition_hash": condition_hash,
                    "shuffle_enabled": shuffle_enabled,
                    "order_seed": trial_seed,
                    "generation_seed": generation.seed,
                    "model_profile_key": model_profile_key(provider, model),
                    "profile_id": sheet.profile_id,
                    "profile_version": sheet.profile_version,
                    "n_items": sheet.n_items,
                    "planned_at_utc": now_utc,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    def _condition_hash(
        self,
        scale: ScaleFile,
        sheet: ScaleSheet,
        language: str,
        provider: str,
        model: str,
        generation: GenerationConfig,
    ) -> str:
        # Deliberately exclude the descriptive condition_mode label.  If a
        # natural and controlled run resolves to the same requested/effective
        # condition, the shared hash makes that overlap detectable instead of
        # hiding it behind two artificial hashes.
        material_hash = None
        try:
            if scale.path.exists():
                material_hash = hashlib.sha256(scale.path.read_bytes()).hexdigest()
        except OSError:
            material_hash = None
        value = {
            "scale_name": scale.name,
            "material_hash": material_hash,
            "language": language,
            "provider": provider,
            "model": model,
            "profile_id": sheet.profile_id,
            "profile_version": sheet.profile_version,
            "generation_config": generation.to_dict(),
            "system_prompt": SYSTEM_PROMPT,
            # A prompt condition changes the prompt itself, so two runs that
            # differ only here are different conditions and must not share a
            # hash.
            "prompt_contract": self.config.prompt_contract,
            "cultural_identity": self.config.cultural_identity,
            "schema": "scale-result-v3" if sheet.is_profiled else "scale-result-v2",
        }
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _execution_hash(self) -> str:
        """Hash operational controls separately from the scientific condition."""

        value = {
            "execution_schema": "execution-v2",
            "order_strategy": self.config.order_strategy,
            "random_seed": self._effective_order_seed(),
            "concurrency": int(self.config.concurrency),
            "scale_concurrency": int(self.config.scale_concurrency),
            "max_retries_per_trial": int(self.config.max_retries),
            "retry_delay": float(self.config.retry_delay),
            "max_group_attempts": int(self.config.max_group_attempts),
            "sample_mode": self.config.sample_mode,
            "network_guard": self.config.network_guard,
            "capture_network": bool(self.config.capture_network),
            "capture_model_catalog": bool(self.config.capture_model_catalog),
        }
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _write_attempt_record(
        self,
        *,
        scale: ScaleFile,
        language: str,
        order_id: int,
        pair_id: Optional[int] = None,
        order_strategy: Optional[str] = None,
        shuffle_seed: Optional[int] = None,
        shuffle_order: Optional[List[int]] = None,
        attempt: int,
        provider: str,
        model: str,
        request_snapshot: Dict[str, Any],
        prompt_sha256: Optional[str] = None,
        raw_response: str,
        response_snapshot: Any,
        model_metadata: Dict[str, Any],
        status: str,
        error: Optional[str] = None,
        condition_hash: Optional[str] = None,
        execution_hash: Optional[str] = None,
        condition_descriptor: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist one structured attempt; failure here never triggers a new API call."""

        attempt_metadata = dict(model_metadata or {})
        attempt_profile = attempt_metadata.pop("model_profile", None)
        attempt_audit = attempt_metadata.get("parameter_audit")
        if isinstance(attempt_audit, dict) and "profile" in attempt_audit:
            attempt_audit = dict(attempt_audit)
            attempt_audit.pop("profile", None)
            attempt_metadata["parameter_audit"] = attempt_audit
        run_part = sanitize_name(self.config.run_id or "adhoc")
        folder = (
            Path(self.config.results_root)
            / run_part
            / sanitize_name(scale.name)
            / ("ch" if language == "ch" else "en")
            / sanitize_name(model)
            / "attempts"
        )
        path = folder / f"trial_{order_id:04d}_attempt_{attempt:02d}.json"
        record = {
            "attempt_schema_version": "attempt-v2",
            "run_id": self.config.run_id or None,
            "scale_name": scale.name,
            "language": language,
            "order_id": order_id,
            "trial_key": f"{scale.name}/{language}/{int(order_id):04d}",
            "pair_id": pair_id,
            "paired_trial_key": (
                f"{scale.name}/{int(pair_id):04d}" if pair_id is not None else None
            ),
            "order_strategy": order_strategy or self.config.order_strategy,
            "shuffle_seed": shuffle_seed,
            "shuffle_order": shuffle_order,
            "attempt": attempt,
            "provider": provider,
            "model": model,
            "condition_mode": self.config.condition_mode,
            "condition_descriptor": condition_descriptor,
            "status": status,
            "error": error,
            "request_snapshot": request_snapshot,
            "response_snapshot": response_snapshot,
            "response_text": raw_response,
            "model_metadata": attempt_metadata,
            "model_profile": attempt_profile,
            "attempt_id": attempt_metadata.get("attempt_id"),
            "client_request_id": attempt_metadata.get("client_request_id"),
            "provider_request_id": attempt_metadata.get("provider_request_id"),
            "network_snapshot_id": attempt_metadata.get("network_snapshot_id"),
            "request_sha256": canonical_hash(request_snapshot),
            "prompt_sha256": prompt_sha256,
            "response_sha256": (
                canonical_hash(response_snapshot) if response_snapshot is not None else None
            ),
            "condition_hash": condition_hash,
            "execution_hash": execution_hash,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            folder.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(path.suffix + ".tmp")
            temp_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temp_path.replace(path)
        except (OSError, TypeError, ValueError) as exc:
            self._ui_log(f"[警告] attempt记录保存失败（不重新调用模型）：{exc}")

    def validate_formal_configuration(self) -> List[str]:
        """Validate all selected model/config pairs before the first request."""

        errors: List[str] = []
        generation = self._effective_generation_config()
        for scale in self.scales:
            cfg = self.config.scale_cfgs.get(scale.name)
            if cfg is None or not cfg.enabled:
                continue
            for language, sheet, count in (
                ("en", scale.en, cfg.en_count),
                ("ch", scale.ch, cfg.ch_count),
            ):
                if sheet is None or count <= 0:
                    continue
                provider, model = self._scale_api(scale.name, language)
                errors.extend(
                    f"{scale.name}/{language}: {message}"
                    for message in validate_generation_config(
                        provider,
                        model,
                        generation,
                        strict=self.config.strict_capabilities,
                    )
                )
        return errors

    def _one_call(
        self,
        scale: ScaleFile,
        sheet: ScaleSheet,
        language: str,
        order_id: int,
        provider: str,
        model: str,
        attempt_guard: Optional[Callable[[], bool]] = None,
    ) -> Path:
        sheet = self._effective_sheet(scale, sheet)
        if sheet.profile_error:
            raise ValueError(
                f"量表专属结构校验失败，已拒绝发送请求：{sheet.profile_error}"
            )
        shuffle_enabled = self._should_shuffle(scale.name, sheet)
        seed = self._trial_seed(scale.name, language, order_id, shuffle_enabled)
        rng = random.Random(seed) if seed is not None else None
        contract = self.config.prompt_contract
        prompt, order, display_plan = build_trial_prompt(
            sheet, language, shuffle_enabled, rng, contract,
            self.config.cultural_identity,
        )
        image_urls = [image.data_url for image in sheet.images]
        cred = get_api_credentials(provider)
        generation = self._effective_generation_config()
        condition_descriptor = build_condition_descriptor(
            self.config.condition_mode,
            generation,
            prompt_contract=self.config.prompt_contract,
            cultural_identity=self.config.cultural_identity,
        )
        condition_hash = self._condition_hash(
            scale, sheet, language, provider, model, generation
        )
        execution_hash = self._execution_hash()
        request_messages = [{"role": "user", "content": prompt}]
        if SYSTEM_PROMPT.strip():
            request_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        request_snapshot = build_chat_request(
            provider,
            model=model,
            messages=request_messages,
            config=generation,
            stream=False,
        )
        # Keep attachment identity without duplicating base64 payloads in every
        # result file.  The actual image bytes still go to the provider request.
        request_snapshot["image_refs"] = [image.metadata() for image in sheet.images]

        last_err: Optional[Exception] = None
        response_snapshot: Any = None
        max_attempts = max(1, int(self.config.max_retries))
        for attempt in range(1, max_attempts + 1):
            if self._stop.is_set():
                raise RuntimeError("用户已停止")
            attempt_id = uuid.uuid4().hex
            request_scope = "\0".join(
                (
                    str(self.config.run_id or "adhoc"),
                    str(scale.name),
                    str(language),
                    str(order_id),
                    str(attempt),
                )
            )
            # HTTP header values must be ASCII-safe. Scale names may be Chinese;
            # hash the identifying scope instead of placing its raw text in a
            # provider-facing X-Client-Request-Id header.
            request_scope_hash = hashlib.sha256(
                request_scope.encode("utf-8")
            ).hexdigest()[:20]
            client_request_id = (
                f"llm-{request_scope_hash}-a{attempt:02d}-"
                f"{uuid.uuid4().hex[:12]}"
            )
            network_snapshot_id = (
                self._run_context.get("network_start_snapshot_id")
                if self._run_context.get("network_start")
                else None
            )
            model_metadata: Dict[str, Any] = {
                "provider": provider,
                "requested_model": model,
                "attempt_id": attempt_id,
                "client_request_id": client_request_id,
                "network_snapshot_id": network_snapshot_id,
                "model_catalog_snapshot_id": self._model_catalog_snapshot_ids.get(provider),
                "account_snapshot_id": self._account_snapshot_ids.get(provider),
                "endpoint": _safe_endpoint(cred.get("base_url")),
                "condition_mode": self.config.condition_mode,
                "condition_descriptor": condition_descriptor,
                "condition_config_sha256": condition_config_hash(generation),
            }
            # Even a failed attempt retains the prepared request and the
            # official documentation profile.
            model_profile = build_model_profile(
                provider,
                model,
                config=generation,
                request_snapshot=request_snapshot,
                model_metadata=model_metadata,
            )
            model_metadata["model_profile"] = model_profile
            response_snapshot = None
            try:
                raw = ""
                if attempt_guard is not None and not attempt_guard():
                    raise AttemptBudgetExhausted("整组 API attempt 预算已用尽")
                proxy_util.apply_proxy_to_env(self._resolved_proxy)
                if self._file_logger:
                    self._file_logger.log(
                        f"开始请求 {scale.name}/{language} #{order_id} "
                        f"provider={provider} model={model} "
                    f"attempt={attempt}/{max_attempts} shuffle={'on' if shuffle_enabled else 'off'} "
                        f"seed={seed} images={len(image_urls)}"
                    )
                request_started = datetime.now(timezone.utc)
                api_result = call_api(
                    provider,
                    prompt,
                    model=model,
                    temperature=generation.temperature,
                    api_key=cred.get("api_key"),
                    base_url=cred.get("base_url"),
                    system_prompt=SYSTEM_PROMPT,
                    image_urls=image_urls,
                    return_metadata=True,
                    generation_config=generation,
                    client_request_id=client_request_id,
                    attempt_id=attempt_id,
                    proxy=self._resolved_proxy,
                )
                # Keep compatibility with tests and third-party callers that
                # monkey-patch call_api and return plain text.
                if isinstance(api_result, dict) and "text" in api_result:
                    raw = str(api_result.get("text") or "")
                    model_metadata = dict(api_result.get("model_metadata") or {})
                    response_snapshot = api_result.get("response_snapshot")
                else:
                    raw = str(api_result or "")
                    model_metadata = {}
                    response_snapshot = None
                model_metadata.setdefault("provider", provider)
                model_metadata.setdefault("requested_model", model)
                model_metadata.setdefault("attempt_id", attempt_id)
                model_metadata.setdefault("client_request_id", client_request_id)
                model_metadata.setdefault("network_snapshot_id", network_snapshot_id)
                model_metadata.setdefault(
                    "model_catalog_snapshot_id",
                    self._model_catalog_snapshot_ids.get(provider),
                )
                model_metadata.setdefault(
                    "account_snapshot_id",
                    self._account_snapshot_ids.get(provider),
                )
                model_metadata.setdefault(
                    "endpoint", _safe_endpoint(cred.get("base_url"))
                )
                model_metadata.setdefault("condition_mode", self.config.condition_mode)
                model_metadata.setdefault("condition_descriptor", condition_descriptor)
                model_metadata.setdefault(
                    "condition_config_sha256", condition_config_hash(generation)
                )
                request_completed = datetime.now(timezone.utc)
                started_utc, started_local = _timestamp_pair(request_started)
                completed_utc, completed_local = _timestamp_pair(request_completed)
                # The runner owns these request-boundary timestamps. Overwrite
                # legacy/provider values so a stale API client cannot preserve
                # a Beijing timestamp under a field named ``*_utc``.
                model_metadata["request_started_at_utc"] = started_utc
                model_metadata["request_started_at_local"] = started_local
                model_metadata["request_completed_at_utc"] = completed_utc
                model_metadata["request_completed_at_local"] = completed_local
                recorded_utc, recorded_local = _normalize_recorded_timestamp(
                    model_metadata, request_completed
                )
                model_metadata["recorded_at_utc"] = recorded_utc
                model_metadata["recorded_at_local"] = recorded_local
                model_profile = build_model_profile(
                    provider,
                    model,
                    config=generation,
                    request_snapshot=request_snapshot,
                    response_snapshot=response_snapshot,
                    returned_model=model_metadata.get("returned_model"),
                    model_metadata=model_metadata,
                )
                model_metadata["model_profile"] = model_profile
                model_metadata["parameter_audit"] = {
                    "configured": generation.to_dict(),
                    "sent": request_snapshot,
                    "model_profile_key": model_profile_key(provider, model),
                    "profile": model_profile,
                    "observed": {
                        "returned_model": model_metadata.get("returned_model"),
                        "finish_reason": model_metadata.get("finish_reason"),
                        "usage": model_metadata.get("usage"),
                        "reasoning_content_present": model_metadata.get(
                            "reasoning_content_present"
                        ),
                    },
                }
                free_slots = display_plan.slots if display_plan else sheet.slots
                parsed = (
                    parse_kohlberg_free_answers(raw, free_slots)
                    if sheet.profile_id == "kohlberg_mji_v1"
                    else parse_free_answers(raw, free_slots, allow_positional_fallback=True)
                )
                # A returned model message is itself a trial observation. Keep
                # complete, partial, refusal, unparsed, and empty responses in
                # the formal result file. Only exceptions raised by the API or
                # transport layer enter the retry path below.
                response_status = classify_model_response(raw, parsed)
                answers = remap_to_original(parsed.answers, order)
                scores = [answer.score if answer is not None else None for answer in answers]
                payload = build_result_payload(
                    scale_name=scale.name,
                    language=language,
                    provider=provider,
                    model=model,
                    temperature=generation.temperature,
                    order_id=order_id,
                    seed=seed,
                    order=order,
                    shuffle_enabled=shuffle_enabled,
                    sheet=sheet,
                    answers_original=answers,
                    parse_result=parsed,
                    raw_response=raw,
                    prompt=prompt,
                    display_plan=display_plan,
                    model_metadata=model_metadata,
                    generation_config=generation.to_dict(),
                    request_snapshot=request_snapshot,
                    response_snapshot=response_snapshot,
                    run_id=self.config.run_id or None,
                    condition_mode=self.config.condition_mode,
                    order_strategy=self.config.order_strategy,
                    pair_id=(
                        order_id if self.config.order_strategy == "paired" else None
                    ),
                    condition_descriptor=condition_descriptor,
                    condition_hash=condition_hash,
                    execution_hash=execution_hash,
                    prompt_contract=contract,
                    cultural_identity=self.config.cultural_identity,
                    response_status=response_status,
                )
                path = self._result_path(
                    scale.name, language, order_id, model, create_parent=True
                )
                payload["preset_id"] = self.config.preset_id
                try:
                    path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    csv_path = path.with_suffix(".csv")
                    self._write_csv(csv_path, payload)
                except (OSError, TypeError, ValueError) as exc:
                    raise StorageError(f"本地结果保存失败：{exc}") from exc
                self._write_attempt_record(
                    scale=scale,
                    language=language,
                    order_id=order_id,
                    pair_id=(
                        order_id if self.config.order_strategy == "paired" else None
                    ),
                    order_strategy=self.config.order_strategy,
                    shuffle_seed=seed,
                    shuffle_order=[index + 1 for index in order],
                    attempt=attempt,
                    provider=provider,
                    model=model,
                    request_snapshot=request_snapshot,
                    prompt_sha256=canonical_hash(prompt),
                    raw_response=raw,
                    response_snapshot=response_snapshot,
                    model_metadata=model_metadata,
                    status=response_status,
                    error=parsed.error,
                    condition_hash=condition_hash,
                    execution_hash=execution_hash,
                    condition_descriptor=condition_descriptor,
                )
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
                model_metadata.update(extract_error_metadata(exc))
                model_metadata["attempt_id"] = attempt_id
                model_metadata["client_request_id"] = client_request_id
                model_metadata["network_snapshot_id"] = network_snapshot_id
                model_metadata["model_profile"] = build_model_profile(
                    provider,
                    model,
                    config=generation,
                    request_snapshot=request_snapshot,
                    response_snapshot=response_snapshot,
                    returned_model=model_metadata.get("returned_model"),
                    model_metadata=model_metadata,
                )
                self._write_attempt_record(
                    scale=scale,
                    language=language,
                    order_id=order_id,
                    pair_id=(
                        order_id if self.config.order_strategy == "paired" else None
                    ),
                    order_strategy=self.config.order_strategy,
                    shuffle_seed=seed,
                    shuffle_order=[index + 1 for index in order],
                    attempt=attempt,
                    provider=provider,
                    model=model,
                    request_snapshot=request_snapshot,
                    prompt_sha256=canonical_hash(prompt),
                    raw_response=raw,
                    response_snapshot=response_snapshot,
                    model_metadata=model_metadata,
                    status="error",
                    error=str(exc),
                    condition_hash=condition_hash,
                    execution_hash=execution_hash,
                    condition_descriptor=condition_descriptor,
                )
                msg = (
                    f"[重试 {attempt}/{max_attempts}] "
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
                        response=raw or None,
                        error=str(exc),
                        seed=seed,
                        shuffle_order=[i + 1 for i in order],
                    )
                if isinstance(exc, StorageError):
                    raise
                if isinstance(exc, AttemptBudgetExhausted):
                    raise
                if attempt < self.config.max_retries and not self._stop.is_set():
                    time.sleep(self.config.retry_delay * attempt)
        raise RuntimeError(
            f"调用失败已达最大重试次数: {scale.name}/{language} #{order_id} "
            f"[{provider}/{model}]: {last_err}"
        )

    @staticmethod
    def _write_csv(path: Path, payload: dict) -> None:
        fieldnames = [
            "index",
            "slot_id",
            "block_id",
            "context_id",
            "section_id",
            "response_type",
            "minimum",
            "maximum",
            "linked_slot_id",
            "question",
            "answer",
            "score",
            "range_text",
            "range_lower",
            "range_upper",
            "range_unit",
            "parse_status",
            "parse_error",
            "mapping_method",
            "answer_present",
            "score_present",
            "trial_status",
            "cultural_identity",
            "trial_raw_response",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item_index, item in enumerate(payload["items"]):
                row = {name: item.get(name, "") for name in fieldnames}
                row["trial_status"] = payload.get("response_status", "")
                row["cultural_identity"] = payload.get("cultural_identity", "none")
                row["trial_raw_response"] = (
                    payload.get("raw_response", "") if item_index == 0 else ""
                )
                writer.writerow(row)

    def _write_run_manifest(self, path: Path, tasks: List[Tuple[Any, ...]]) -> None:
        generation = self._effective_generation_config()
        model_profiles: Dict[str, Dict[str, Any]] = {}
        account_snapshots: Dict[str, Dict[str, Any]] = {}
        model_catalogs: Dict[str, Dict[str, Any]] = {}
        for _scale, _sheet, _language, _count, provider, model in tasks:
            key = model_profile_key(provider, model)
            if key not in model_profiles:
                model_profiles[key] = build_model_profile(
                    provider,
                    model,
                    config=generation,
                )
            if provider not in account_snapshots:
                credential = get_api_credentials(provider)
                account_snapshot = build_account_snapshot(
                    provider, credential.get("api_key")
                )
                account_snapshot["snapshot_id"] = canonical_hash(account_snapshot)
                account_snapshots[provider] = account_snapshot
                self._account_snapshot_ids[provider] = account_snapshot["snapshot_id"]
                if self.config.capture_model_catalog:
                    catalog = fetch_model_catalog(
                        provider,
                        api_key=credential.get("api_key"),
                        base_url=credential.get("base_url"),
                        proxy=self._resolved_proxy,
                    )
                    model_catalogs[provider] = catalog
                    self._model_catalog_snapshot_ids[provider] = canonical_hash(catalog)
        for key, profile in model_profiles.items():
            provider = key.split("/", 1)[0]
            snapshot_id = self._model_catalog_snapshot_ids.get(provider)
            if snapshot_id:
                profile.setdefault("identity", {})[
                    "model_catalog_snapshot_id"
                ] = snapshot_id
            account_snapshot_id = self._account_snapshot_ids.get(provider)
            if account_snapshot_id:
                profile.setdefault("identity", {})[
                    "account_snapshot_id"
                ] = account_snapshot_id
        run_context = self._run_context or {
            "schema_version": "run-context-v1",
            "run_id": self.config.run_id or None,
            "software": build_software_snapshot(),
            "network_start": None,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest: Dict[str, Any] = {
            # Keep the manifest version stable for existing analysis scripts;
            # model_profiles is additive and has its own schema_version.
            "manifest_version": "run-manifest-v1",
            "application_version": APP_VERSION,
            "run_id": self.config.run_id,
            "preset_id": self.config.preset_id or "custom",
            "condition_mode": self.config.condition_mode,
            # The prompt condition is part of the scientific condition.
            "prompt_contract": self.config.prompt_contract,
            "cultural_identity": self.config.cultural_identity,
            "prompt_config": {
                "contract": self.config.prompt_contract,
                "cultural_identity": self.config.cultural_identity,
            },
            "order_strategy": self.config.order_strategy,
            "order_design": {
                "mode": self.config.order_strategy,
                "master_seed": self._effective_order_seed(),
                "configured_master_seed": self.config.random_seed,
                "pair_key_fields": ["scale_name", "order_id"]
                if self.config.order_strategy == "paired"
                else [],
                "languages_share_order_seed": self.config.order_strategy == "paired",
            },
            "condition_descriptor": build_condition_descriptor(
                self.config.condition_mode,
                generation,
                prompt_contract=self.config.prompt_contract,
                cultural_identity=self.config.cultural_identity,
            ),
            "condition_config_sha256": condition_config_hash(generation),
            "condition_analysis": {
                "primary_estimand": "within_model_language_scale",
                "paired_key_fields": [
                    "scale_name",
                    "language",
                    "provider",
                    "requested_model",
                    "trial_key",
                ],
                "natural_and_controlled_are_independent": False,
                "effective_equivalence_requires": [
                    "condition_hash",
                    "request_sha256",
                    "returned_model",
                    "provider_response_metadata",
                ],
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "created_at_local": datetime.now(timezone.utc).astimezone(SHANGHAI_TZ).isoformat(),
            "generation_config": generation.to_dict(),
            "model_profiles": model_profiles,
            "account_snapshots": account_snapshots,
            "model_catalogs": model_catalogs,
            "run_context": run_context,
            "network_guard": self.config.network_guard,
            "status": "started",
            "temperature_legacy": generation.temperature,
            "strict_capabilities": self.config.strict_capabilities,
            "concurrency": self.config.concurrency,
            "scale_concurrency": self.config.scale_concurrency,
            "max_retries_per_trial": self.config.max_retries,
            "max_group_attempts": self.config.max_group_attempts,
            "sample_mode": self.config.sample_mode,
            "random_seed": self.config.random_seed,
            "plan_path": display_path(self._plan_path()),
            "execution_config": {
                "order_strategy": self.config.order_strategy,
                "random_seed": self._effective_order_seed(),
                "concurrency": int(self.config.concurrency),
                "scale_concurrency": int(self.config.scale_concurrency),
                "max_retries_per_trial": int(self.config.max_retries),
                "retry_delay": float(self.config.retry_delay),
                "max_group_attempts": int(self.config.max_group_attempts),
                "sample_mode": self.config.sample_mode,
                "network_guard": self.config.network_guard,
                "capture_network": bool(self.config.capture_network),
                "capture_model_catalog": bool(self.config.capture_model_catalog),
            },
            "tasks": [
                {
                    "scale": scale.name,
                    "language": language,
                    "count": count,
                    "provider": provider,
                    "model": model,
                    "condition_mode": self.config.condition_mode,
                    "condition_config_sha256": condition_config_hash(generation),
                    "profile_id": sheet.profile_id,
                    "profile_version": sheet.profile_version,
                    "n_items": sheet.n_items,
                    "mji_form": (
                        sheet.profile_version.rsplit("-form-", 1)[-1]
                        if sheet.profile_id == "kohlberg_mji_v1"
                        and "-form-" in sheet.profile_version
                        else "all"
                    ),
                    "shuffle_enabled": self._should_shuffle(scale.name, sheet),
                    "shuffle_pair_compatible": paired_shuffle_compatible(
                        scale.en, scale.ch
                    ),
                    "model_profile_key": model_profile_key(provider, model),
                    "source_path": display_path(scale.path),
                    "source_file_sha256": file_sha256(scale.path),
                }
                for scale, sheet, language, count, provider, model in tasks
            ],
        }
        execution = manifest["execution_config"]
        execution_canonical = json.dumps(
            {"execution_schema": "execution-v2", **execution},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest["execution_hash"] = hashlib.sha256(
            execution_canonical.encode("utf-8")
        ).hexdigest()
        manifest["manifest_hash"] = _manifest_hash(manifest)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self._manifest_path = path

    def _finalize_run_manifest(self, status: str) -> None:
        """Add completion/network observations without breaking a started manifest."""

        path = self._manifest_path
        if path is None or not path.exists():
            return
        if self.config.capture_network:
            self._network_end = build_network_snapshot(
                proxy=self._resolved_proxy,
                include_public_ip=True,
            )
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            context = dict(manifest.get("run_context") or {})
            context["network_end"] = self._network_end
            context["network_end_snapshot_id"] = (
                self._network_end or {}
            ).get("snapshot_id")
            end_comparison = compare_network_snapshots(
                context.get("network_start"), self._network_end
            )
            initial_comparison = context.get("network_start_comparison") or {}
            changed = bool(
                initial_comparison.get("network_changed")
                or end_comparison.get("network_changed")
            )
            context["network_comparison"] = {
                "start_confirmation": initial_comparison,
                "end": end_comparison,
                "network_changed": changed,
                "status": "changed" if changed else end_comparison.get("status"),
            }
            context["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["run_context"] = context
            manifest["status"] = status
            manifest["completed_at_utc"] = context["completed_at_utc"]
            if context["network_comparison"].get("network_changed"):
                manifest["network_warning"] = (
                    "运行开始与结束的出口网络观测不一致；请将网络条件作为独立运行条件处理。"
                )
            manifest["manifest_hash"] = _manifest_hash(manifest)
            temp_path = path.with_suffix(path.suffix + ".tmp")
            temp_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temp_path.replace(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._ui_log(f"[警告] 运行结束快照更新失败：{exc}")

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
    ) -> bool:
        sheet = self._effective_sheet(scale, sheet)
        task_key = f"{scale.name}/{language}"
        self._ui_log(
            f"—— 开始 {task_key} | API={provider} | 模型={model}，"
            f"目标保存 {count} 个试次，并发={self.config.concurrency} ——"
        )
        if sheet.images:
            self._ui_log(
                f"[附图] {task_key}: 将随每次请求发送 {len(sheet.images)} 张嵌入图片；"
                "请确认所选模型支持视觉输入。"
            )

        success = 0
        submitted_trials = 0
        order_id = 1
        while self._result_path(scale.name, language, order_id, model).exists():
            order_id += 1

        id_lock = threading.Lock()
        next_order_id = order_id
        state_lock = threading.Lock()
        budget_lock = threading.Lock()
        api_attempts_used = 0

        max_api_attempts = int(self.config.max_group_attempts or 0)
        if max_api_attempts <= 0:
            max_api_attempts = count * max(1, int(self.config.max_retries))
        max_api_attempts = max(count, max_api_attempts)
        # The requested count is the complete trial plan.  A refusal,
        # partial answer, unparsed answer, or exhausted transport retry keeps
        # its original trial ID and is never replaced by a new trial.
        max_trial_submissions = count

        def attempt_guard() -> bool:
            nonlocal api_attempts_used
            with budget_lock:
                if api_attempts_used >= max_api_attempts:
                    return False
                api_attempts_used += 1
                return True

        def submit_job(executor: ThreadPoolExecutor):
            nonlocal next_order_id
            with id_lock:
                oid = next_order_id
                next_order_id += 1

            def job() -> Tuple[bool, Optional[Path], Optional[str], int]:
                try:
                    path = self._one_call(
                        scale,
                        sheet,
                        language,
                        oid,
                        provider,
                        model,
                        attempt_guard,
                    )
                    return True, path, None, oid
                except Exception as exc:
                    return False, None, str(exc), oid

            return executor.submit(job)

        workers = max(1, int(self.config.concurrency))
        plan_count = max_trial_submissions
        planned_ids = list(range(next_order_id, next_order_id + plan_count))
        self._write_trial_plan(
            scale=scale,
            sheet=sheet,
            language=language,
            provider=provider,
            model=model,
            order_ids=planned_ids,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = set()
            while True:
                if self._stop.is_set():
                    self._ui_log("已停止。")
                    for fut in pending:
                        fut.cancel()
                    return False

                while (
                    len(pending) < workers
                    and submitted_trials < count
                    and submitted_trials < max_trial_submissions
                    and not self._stop.is_set()
                ):
                    pending.add(submit_job(executor))
                    submitted_trials += 1

                if not pending:
                    break

                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    ok, path, err, oid = fut.result()
                    if ok and path is not None:
                        with state_lock:
                            counted = success < count
                            if counted:
                                success += 1
                            s = success
                        with self._progress_lock:
                            if counted:
                                grand_done[0] += 1
                            g = grand_done[0]
                        self._ui_log(
                            f"[已保存 {s}/{count}] {task_key} [{provider}/{model}] "
                            f"#{oid} -> {path.name}"
                        )
                        self.progress(task_key, s, count)
                        self.progress("__all__", g, grand_total)
                    else:
                        if err and "用户已停止" in err:
                            self._ui_log("已停止。")
                            return False
                        self._ui_log(
                            f"[失败] {task_key} #{oid}: {err}；该试次保留为失败，不生成替代试次。"
                        )

                with state_lock:
                    if success >= count:
                        break

                if not pending and submitted_trials >= max_trial_submissions:
                    self._ui_log(
                        f"[停止] {task_key} 已提交固定计划 trial {submitted_trials} 个；"
                        f"已保存模型响应 {success}/{count}，失败试次不补替。"
                    )
                    break

        self._ui_log(
            f"—— 完成 {task_key}：已保存 {success}/{count}，"
            f"提交 trial {submitted_trials}/{max_trial_submissions}，"
            f"API attempt {api_attempts_used}/{max_api_attempts} ——"
        )
        return success >= count

    def run(self) -> None:
        logs_root = Path(self.config.logs_root)
        logs_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.config.run_id:
            self.config.run_id = f"run_{stamp}_{uuid.uuid4().hex[:8]}"
        if self.config.order_strategy != "fixed" and self.config.random_seed is None:
            self.config.random_seed = self._effective_random_seed()
        log_path = logs_root / f"{sanitize_name(self.config.run_id)}.log"
        self._file_logger = RunFileLogger(log_path)
        self._ui_log(f"详细日志文件: {display_path(log_path)}")

        self._resolved_proxy = proxy_util.detect_system_proxy()
        proxy = proxy_util.apply_proxy_to_env(self._resolved_proxy)
        proxy_identity = sanitize_proxy(self._resolved_proxy)
        self._ui_log(
            f"系统代理: {proxy_identity['display']}"
            if proxy
            else "未使用代理，直连 API"
        )
        self._run_context = build_run_context(
            run_id=self.config.run_id,
            proxy=self._resolved_proxy,
            include_public_ip=bool(self.config.capture_network),
        )
        if self.config.capture_network and self.config.network_guard == "stop":
            # Confirm the initial egress twice before any model request. This
            # catches a rotating proxy early without querying an IP service for
            # every concurrent request.
            confirmation = build_network_snapshot(
                proxy=self._resolved_proxy,
                include_public_ip=True,
                timeout=1.5,
            )
            self._run_context["network_start_confirmation"] = confirmation
            initial_comparison = compare_network_snapshots(
                self._run_context.get("network_start"), confirmation
            )
            self._run_context["network_start_comparison"] = initial_comparison
            if initial_comparison.get("network_changed"):
                self._ui_log(
                    "[阻止] 运行开始时出口网络观测不稳定；请固定代理/IP后重新运行。"
                )
                self._file_logger.close_note("启动前网络不稳定")
                return
        self._ui_log(
            f"条件模式={self.config.condition_mode}, "
            f"文化身份={self.config.cultural_identity}, "
            f"温度={self._effective_generation_config().temperature}, "
            f"量表并行={self.config.scale_concurrency}, 单量表并发={self.config.concurrency}"
        )
        condition_descriptor = build_condition_descriptor(
            self.config.condition_mode,
            self._effective_generation_config(),
            prompt_contract=self.config.prompt_contract,
            cultural_identity=self.config.cultural_identity,
        )
        if is_free_prompt_contract(self.config.prompt_contract):
            self._ui_log(
                "按量表原文发送，不要求模型使用固定回答格式；无法逐题拆分时仍保存完整原始回答。"
            )
        if condition_descriptor["interpretation"] == "natural_with_explicit_overrides":
            self._ui_log(
                "[警告] 当前标记为自然模式，但存在显式生成参数；"
                "分析时请按自定义/受控条件解释。"
            )
        elif condition_descriptor["interpretation"] == "controlled_without_explicit_settings":
            self._ui_log(
                "[警告] 当前标记为受控模式，但没有显式生成参数；"
                "实际可能仍由平台默认值决定，请应用受控预设后再正式运行。"
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
                self.config.scale_cfgs[scale.name] = cfg
            if not cfg.enabled:
                continue
            if scale.en and cfg.en_count > 0:
                provider, model = self._scale_api(scale.name, "en")
                if scale.en.profile_error:
                    self._ui_log(
                        f"[跳过] {scale.name}/en: 专属量表结构校验失败：{scale.en.profile_error}"
                    )
                elif not provider or not model:
                    self._ui_log(f"[跳过] {scale.name}/en: 未指定 API 提供方或模型")
                else:
                    tasks.append(
                        (
                            scale,
                            self._effective_sheet(scale, scale.en),
                            "en",
                            cfg.en_count,
                            provider,
                            model,
                        )
                    )
            if scale.ch and cfg.ch_count > 0:
                provider, model = self._scale_api(scale.name, "ch")
                if scale.ch.profile_error:
                    self._ui_log(
                        f"[跳过] {scale.name}/ch: 专属量表结构校验失败：{scale.ch.profile_error}"
                    )
                elif not provider or not model:
                    self._ui_log(f"[跳过] {scale.name}/ch: 未指定 API 提供方或模型")
                else:
                    tasks.append(
                        (
                            scale,
                            self._effective_sheet(scale, scale.ch),
                            "ch",
                            cfg.ch_count,
                            provider,
                            model,
                        )
                    )

        if not tasks:
            self._ui_log("没有待执行的任务，请检查量表勾选、调用次数与 API 设置。")
            self._file_logger.close_note("无任务，结束")
            return

        capability_errors = self.validate_formal_configuration()
        if capability_errors:
            self._ui_log("当前模型或参数设置无法运行，未发送任何模型请求：")
            for error in capability_errors:
                self._ui_log(f"  - {error}")
            self._file_logger.close_note("思考模式校验失败")
            return

        manifest_path = logs_root / f"{sanitize_name(self.config.run_id)}.manifest.json"
        try:
            self._write_run_manifest(manifest_path, tasks)
            self._ui_log(f"实验配置快照: {display_path(manifest_path)}")
        except OSError as exc:
            self._ui_log(f"无法保存实验配置快照，已停止且未发送请求：{exc}")
            self._file_logger.close_note("配置快照失败")
            return

        for scale, _sheet, language, count, provider, model in tasks:
            self._ui_log(
                f"计划任务: {scale.name}/{language} x{count} -> {provider}/{model} "
                f"| 题项乱序={'开' if self._should_shuffle(scale.name, _sheet) else '关'} "
                f"| 附图={len(_sheet.images)}"
            )

        grand_total = sum(t[3] for t in tasks)
        grand_done = [0]
        scale_tasks: Dict[str, List[Tuple[ScaleFile, ScaleSheet, str, int, str, str]]] = {}
        for task in tasks:
            scale_tasks.setdefault(task[0].name, []).append(task)
        scale_workers = min(int(self.config.scale_concurrency), len(scale_tasks))
        self._ui_log(
            f"开始执行，共 {grand_total} 个试次保存目标；"
            f"同时运行 {scale_workers} 个量表，每个量表内 API 并发 {self.config.concurrency}。"
        )

        def run_scale(
            scale_name: str,
            language_tasks: List[Tuple[ScaleFile, ScaleSheet, str, int, str, str]],
        ) -> bool:
            scale_complete = True
            for scale, sheet, language, count, provider, model in language_tasks:
                if self._stop.is_set():
                    return False
                group_complete = self._run_group(
                    scale, sheet, language, count, provider, model,
                    grand_done, grand_total,
                )
                scale_complete = group_complete and scale_complete
            return scale_complete

        all_groups_complete = True
        with ThreadPoolExecutor(max_workers=max(1, scale_workers)) as executor:
            pending = {
                executor.submit(run_scale, scale_name, language_tasks): scale_name
                for scale_name, language_tasks in scale_tasks.items()
            }
            while pending:
                if self._stop.is_set():
                    for future in pending:
                        future.cancel()
                    break
                done, _ = wait(
                    pending,
                    timeout=0.25,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    scale_name = pending.pop(future)
                    try:
                        all_groups_complete = future.result() and all_groups_complete
                    except Exception as exc:
                        all_groups_complete = False
                        self._ui_log(f"[错误] 量表 {scale_name} 执行失败：{exc}")

        if self._stop.is_set():
            self._ui_log("运行被用户停止。")
            self._file_logger.close_note("用户停止")
            final_status = "stopped"
        elif all_groups_complete:
            self._ui_log("全部任务完成。")
            self._file_logger.close_note("全部完成")
            final_status = "completed"
        else:
            self._ui_log("任务结束，但部分量表未达到目标保存试次数；请查看 manifest、plan 和 attempt 记录。")
            self._file_logger.close_note("部分未完成")
            final_status = "partial"
        try:
            answer_path = export_trials(self._plan_path().parent)
            self._ui_log(f"试次完整回答 Excel: {display_path(answer_path)}")
        except Exception as exc:
            self._ui_log(f"完整回答 Excel 导出失败，JSON/CSV 仍保留：{exc}")
        self._finalize_run_manifest(final_status)
