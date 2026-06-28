from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pipeline_support import ensure_dir, resolve_path


ADAPTIVE_TOP_K_ARTIFACT_LABEL = "cutoff_topk"
PIPELINE_I_ADAPTIVE_TOP_K_MIN_CUTOFF = 0.0
PIPELINE_I_ADAPTIVE_TOP_K_MAX_CUTOFF = 0.5
GLOBAL_CUTOFF_MODE = "global"

ThresholdSpec = Dict[str, Any]
RawThreshold = Union[None, int, float, str, Dict[str, Any]]


@dataclass(frozen=True)
class PipelinePaths:
    outputs_root: Path
    experiment_root: Path
    generated_root: Path
    program_root: Path
    compiled_root: Path
    inputs_root: Path
    staged_experiments_path: Path
    compile_manifest_path: Path
    inference_runs_path: Path
    inference_manifest_path: Path
    visualization_root: Path

    def to_json_dict(self) -> Dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def ensure_compile_dirs(self) -> None:
        ensure_dir(self.program_root)
        ensure_dir(self.compiled_root)

    def ensure_stage_dirs(self) -> None:
        ensure_dir(self.experiment_root)
        ensure_dir(self.inputs_root)

    def ensure_inference_dirs(self) -> None:
        ensure_dir(self.experiment_root)

    def ensure_visualization_dirs(self) -> None:
        ensure_dir(self.visualization_root)


@dataclass(frozen=True)
class PipelineContext:
    paths_cfg: Dict[str, Any]
    inference_cfg: Dict[str, Any]
    show_progress: bool
    paths: PipelinePaths


def threshold_label(cutoff: Optional[float]) -> str:
    return "exact" if cutoff is None else f"cutoff_{str(cutoff).replace('.', 'p')}"


def _label_float(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _normalise_cutoff_search_config(
    config: Dict[str, Any],
    threshold: Dict[str, Any],
) -> Dict[str, Any]:
    inference_cfg = config.get("inference", {}) or {}
    global_cfg = inference_cfg.get("adaptive_top_k", {}) or {}
    local_cfg = threshold.get("cutoff_search", {}) or {}
    if not isinstance(global_cfg, dict):
        raise ValueError("inference.adaptive_top_k must be a mapping when provided.")
    if not isinstance(local_cfg, dict):
        name = threshold.get("name", threshold.get("label"))
        raise ValueError(f"cutoff_search for threshold {name!r} must be a mapping.")

    def value(key: str, default: Any) -> Any:
        return local_cfg.get(key, global_cfg.get(key, default))

    def integer(key: str, default: int) -> int:
        raw_value = value(key, default)
        try:
            return int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"inference.adaptive_top_k.{key} must be an integer, got {raw_value!r}"
            ) from exc

    def number(key: str, default: float) -> float:
        raw_value = value(key, default)
        try:
            return float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"inference.adaptive_top_k.{key} must be numeric, got {raw_value!r}"
            ) from exc

    probe_experiments = integer("probe_experiments", integer("probe_cases", 20))
    max_iterations = integer("max_iterations", 14)
    tolerance = number("tolerance", 0.02)
    min_cutoff = number("min_cutoff", PIPELINE_I_ADAPTIVE_TOP_K_MIN_CUTOFF)
    max_cutoff = number("max_cutoff", PIPELINE_I_ADAPTIVE_TOP_K_MAX_CUTOFF)

    if probe_experiments <= 0:
        raise ValueError("inference.adaptive_top_k.probe_experiments must be positive.")
    if max_iterations <= 0:
        raise ValueError("inference.adaptive_top_k.max_iterations must be positive.")
    if tolerance < 0.0:
        raise ValueError("inference.adaptive_top_k.tolerance must be non-negative.")
    if not (
        PIPELINE_I_ADAPTIVE_TOP_K_MIN_CUTOFF
        <= min_cutoff
        <= max_cutoff
        <= PIPELINE_I_ADAPTIVE_TOP_K_MAX_CUTOFF
    ):
        raise ValueError(
            "Pipeline I adaptive top-k search bounds must satisfy "
            f"{PIPELINE_I_ADAPTIVE_TOP_K_MIN_CUTOFF:g} <= min_cutoff <= max_cutoff <= "
            f"{PIPELINE_I_ADAPTIVE_TOP_K_MAX_CUTOFF:g}."
        )

    return {
        "probe_experiments": probe_experiments,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "min_cutoff": min_cutoff,
        "max_cutoff": max_cutoff,
    }


def _normalise_threshold(config: Dict[str, Any], raw: RawThreshold) -> ThresholdSpec:
    if raw is None:
        return {
            "threshold_label": "exact",
            "artifact_threshold_label": threshold_label(None),
            "cutoff": None,
            "compile_cutoff": None,
            "adaptive_top_k": False,
            "posterior_mass_target": None,
            "cutoff_search": None,
        }

    if isinstance(raw, dict):
        item = dict(raw)
        name = item.get("name", item.get("label"))
        adaptive_requested = bool(item.get("adaptive_top_k", False)) or (
            item.get("posterior_mass_target") is not None
        )
        raw_cutoff = item.get("top_k_cutoff", item.get("cutoff"))
        if isinstance(raw_cutoff, str) and raw_cutoff.strip().lower() in {"auto", "adaptive"}:
            adaptive_requested = True
            raw_cutoff = 0.0
        if adaptive_requested and raw_cutoff is None:
            raw_cutoff = 0.0

        cutoff = None if raw_cutoff is None else float(raw_cutoff)
        if cutoff is not None and not 0.0 <= cutoff <= 1.0:
            raise ValueError(
                f"Every topKCutoff value must be between 0 and 1. Invalid value: {raw_cutoff}"
            )

        posterior_mass_target = None
        cutoff_search = None
        if adaptive_requested:
            global_cfg = (config.get("inference", {}) or {}).get("adaptive_top_k", {}) or {}
            if not isinstance(global_cfg, dict):
                raise ValueError("inference.adaptive_top_k must be a mapping when provided.")
            posterior_mass_target = float(
                item.get("posterior_mass_target", global_cfg.get("posterior_mass_target", 0.8))
            )
            if not 0.0 < posterior_mass_target <= 1.0:
                raise ValueError(
                    f"posterior_mass_target must be in (0, 1], got {posterior_mass_target}"
                )
            cutoff_search = _normalise_cutoff_search_config(config, item)
            if not name:
                name = f"approx_mass_{_label_float(posterior_mass_target)}"
        elif not name:
            name = threshold_label(cutoff)

        return {
            "threshold_label": str(name),
            "artifact_threshold_label": (
                ADAPTIVE_TOP_K_ARTIFACT_LABEL if adaptive_requested else threshold_label(cutoff)
            ),
            "cutoff": cutoff,
            "compile_cutoff": cutoff,
            "adaptive_top_k": adaptive_requested,
            "posterior_mass_target": posterior_mass_target,
            "cutoff_search": cutoff_search,
        }

    cutoff = float(raw)
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError(f"Every topKCutoff value must be between 0 and 1. Invalid value: {raw}")
    return {
        "threshold_label": threshold_label(cutoff),
        "artifact_threshold_label": threshold_label(cutoff),
        "cutoff": cutoff,
        "compile_cutoff": cutoff,
        "adaptive_top_k": False,
        "posterior_mass_target": None,
        "cutoff_search": None,
    }


def get_thresholds(config: Dict[str, Any]) -> List[ThresholdSpec]:
    raw_thresholds = list(
        config["inference"].get("approximation_thresholds", [None, 0.001, 0.01, 0.05])
    )
    thresholds = [_normalise_threshold(config, raw) for raw in raw_thresholds]
    labels = [threshold_spec_label(threshold) for threshold in thresholds]
    if len(labels) != len(set(labels)):
        duplicate = next(label for label in labels if labels.count(label) > 1)
        raise ValueError(f"Duplicate inference threshold label: {duplicate}")
    return thresholds


def threshold_spec_label(threshold: ThresholdSpec) -> str:
    return str(threshold["threshold_label"])


def threshold_spec_artifact_label(threshold: ThresholdSpec) -> str:
    return str(
        threshold.get("artifact_threshold_label")
        or threshold_label(threshold_spec_compile_cutoff(threshold))
    )


def threshold_spec_compile_cutoff(threshold: ThresholdSpec) -> Optional[float]:
    cutoff = threshold.get("compile_cutoff", threshold.get("cutoff"))
    return None if cutoff is None else float(cutoff)


def threshold_spec_runtime_seed_cutoff(threshold: ThresholdSpec) -> Optional[float]:
    cutoff = threshold.get("cutoff", threshold.get("compile_cutoff"))
    return None if cutoff is None else float(cutoff)


def threshold_spec_for_json(threshold: ThresholdSpec) -> Dict[str, Any]:
    return {
        "threshold_label": threshold_spec_label(threshold),
        "artifact_threshold_label": threshold_spec_artifact_label(threshold),
        "cutoff": threshold_spec_runtime_seed_cutoff(threshold),
        "compile_cutoff": threshold_spec_compile_cutoff(threshold),
        "adaptive_top_k": bool(threshold.get("adaptive_top_k", False)),
        "posterior_mass_target": threshold.get("posterior_mass_target"),
        "cutoff_search": threshold.get("cutoff_search"),
    }


def normalize_cutoff_mode(mode: Any) -> str:
    value = str(mode).strip().lower()
    if value not in {"local", GLOBAL_CUTOFF_MODE}:
        raise ValueError(f"Unsupported cutoff mode: {mode}")
    return value


def get_term_count_bounds(config: Dict[str, Any]) -> Tuple[int, int]:
    inference_cfg = config["inference"]
    terms_min = int(inference_cfg.get("terms_per_sum_min", 1))
    terms_max = int(inference_cfg.get("terms_per_sum_max", 4))
    if terms_min < 1 or terms_max < terms_min:
        raise ValueError(f"Invalid term count bounds: min={terms_min}, max={terms_max}")
    return terms_min, terms_max


def get_configured_term_counts(config: Dict[str, Any]) -> List[int]:
    terms_min, terms_max = get_term_count_bounds(config)
    return list(range(terms_min, terms_max + 1))


def build_pipeline_context(config: Dict[str, Any]) -> PipelineContext:
    paths_cfg = config["paths"]
    inference_cfg = config["inference"]
    outputs_root = resolve_path(config, paths_cfg.get("outputs_root", "./outputs"))
    experiment_root = outputs_root / "spll_experiments"
    generated_root = experiment_root / "generated"
    paths = PipelinePaths(
        outputs_root=outputs_root,
        experiment_root=experiment_root,
        generated_root=generated_root,
        program_root=generated_root / "spll_programs",
        compiled_root=generated_root / "compiled_python",
        inputs_root=experiment_root / "inputs",
        staged_experiments_path=experiment_root / "staged_experiments.json",
        compile_manifest_path=experiment_root / "compile_manifest.json",
        inference_runs_path=experiment_root / "inference_runs.json",
        inference_manifest_path=experiment_root / "inference_manifest.json",
        visualization_root=experiment_root / "visualization",
    )
    return PipelineContext(
        paths_cfg=paths_cfg,
        inference_cfg=inference_cfg,
        show_progress=bool(inference_cfg.get("show_progress", True)),
        paths=paths,
    )
