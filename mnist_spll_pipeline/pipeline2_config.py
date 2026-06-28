from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from pipeline1_config import ADAPTIVE_TOP_K_ARTIFACT_LABEL
from pipeline_support import ensure_dir, resolve_path


@dataclass(frozen=True)
class TrainingPaths:
    root: Path
    config_used_path: Path
    data_root: Path
    split_manifest_path: Path
    schedules_root: Path
    schedule_previews_root: Path
    initial_checkpoints_root: Path
    generated_root: Path
    program_root: Path
    compiled_root: Path
    runs_root: Path
    visualization_root: Path
    tables_root: Path
    figures_root: Path
    figures_main_text_root: Path
    figures_appendix_root: Path

    def ensure_prepare_dirs(self) -> None:
        for path in (
            self.root,
            self.schedules_root,
            self.schedule_previews_root,
            self.initial_checkpoints_root,
        ):
            ensure_dir(path)

    def ensure_compile_dirs(self) -> None:
        for path in (self.root, self.program_root, self.compiled_root):
            ensure_dir(path)

    def ensure_training_dirs(self) -> None:
        for path in (self.root, self.runs_root):
            ensure_dir(path)

    def ensure_visualization_dirs(self) -> None:
        for path in (
            self.visualization_root,
            self.tables_root,
            self.figures_root,
            self.figures_main_text_root,
            self.figures_appendix_root,
        ):
            ensure_dir(path)


def training_paths(config: Dict[str, Any]) -> TrainingPaths:
    paths_cfg = config.get("paths", {})
    root = resolve_path(config, paths_cfg.get("output_root", "./outputs/spll_training"))
    generated_root = root / "generated"
    visualization_root = root / "visualization"
    schedules_root = root / "schedules"
    figures_root = visualization_root / "figures"
    return TrainingPaths(
        root=root,
        config_used_path=root / "config_used.yaml",
        data_root=resolve_path(config, paths_cfg.get("data_root", "./data")),
        split_manifest_path=root / "data_split_manifest.json",
        schedules_root=schedules_root,
        schedule_previews_root=schedules_root / "previews",
        initial_checkpoints_root=root / "initial_checkpoints",
        generated_root=generated_root,
        program_root=generated_root / "spll_programs",
        compiled_root=generated_root / "compiled_python",
        runs_root=root / "runs",
        visualization_root=visualization_root,
        tables_root=visualization_root / "tables",
        figures_root=figures_root,
        figures_main_text_root=figures_root / "main_text",
        figures_appendix_root=figures_root / "appendix",
    )


def get_experiments(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    experiments = config.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Pipeline II config must define a non-empty experiments list.")

    normalized: List[Dict[str, Any]] = []
    default_max_steps = int(config.get("training", {}).get("max_steps", 1_000_000))
    for raw in experiments:
        if not isinstance(raw, dict):
            raise ValueError("Each experiment entry must be a mapping.")
        if not raw.get("enabled", True):
            continue
        n_terms = int(raw["n_terms"])
        if n_terms < 1:
            raise ValueError(f"n_terms must be >= 1, got {n_terms}")
        item = dict(raw)
        item["n_terms"] = n_terms
        item["max_steps"] = int(raw.get("max_steps", default_max_steps))
        normalized.append(item)

    if not normalized:
        raise ValueError("All Pipeline II experiments are disabled.")
    return normalized


def get_seeds(config: Dict[str, Any]) -> List[int]:
    seeds = config.get("seeds", [int(config.get("seed", 42))])
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list.")
    return [int(seed) for seed in seeds]


def _as_float(value: Any, *, field_name: str, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc


def _as_int(value: Any, *, field_name: str, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc


def _normalize_cutoff_search_config(
    config: Dict[str, Any],
    raw_mode: Dict[str, Any],
) -> Dict[str, Any]:
    global_cfg = config.get("adaptive_top_k", {}) or {}
    mode_cfg = raw_mode.get("cutoff_search", {}) or {}
    if not isinstance(global_cfg, dict):
        raise ValueError("adaptive_top_k must be a mapping when provided.")
    if not isinstance(mode_cfg, dict):
        raise ValueError(f"cutoff_search for mode {raw_mode.get('name')!r} must be a mapping.")

    def value(key: str, default: Any) -> Any:
        return mode_cfg.get(key, global_cfg.get(key, default))

    probe_cases = _as_int(
        value("probe_cases", 20),
        field_name="adaptive_top_k.probe_cases",
        default=20,
    )
    max_iterations = _as_int(
        value("max_iterations", 14),
        field_name="adaptive_top_k.max_iterations",
        default=14,
    )
    tolerance = _as_float(
        value("tolerance", 0.02),
        field_name="adaptive_top_k.tolerance",
        default=0.02,
    )
    min_cutoff = _as_float(
        value("min_cutoff", 0.0),
        field_name="adaptive_top_k.min_cutoff",
        default=0.0,
    )
    max_cutoff = _as_float(
        value("max_cutoff", 1.0),
        field_name="adaptive_top_k.max_cutoff",
        default=1.0,
    )

    if probe_cases <= 0:
        raise ValueError("adaptive_top_k.probe_cases must be positive.")
    if max_iterations <= 0:
        raise ValueError("adaptive_top_k.max_iterations must be positive.")
    if tolerance < 0.0:
        raise ValueError("adaptive_top_k.tolerance must be non-negative.")
    if not 0.0 <= min_cutoff <= max_cutoff <= 1.0:
        raise ValueError(
            "adaptive_top_k min_cutoff/max_cutoff must satisfy "
            "0 <= min_cutoff <= max_cutoff <= 1."
        )

    return {
        "probe_cases": probe_cases,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "min_cutoff": min_cutoff,
        "max_cutoff": max_cutoff,
    }


def get_inference_modes(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    modes = config.get("inference_modes", [])
    if not isinstance(modes, list) or not modes:
        raise ValueError("Pipeline II config must define a non-empty inference_modes list.")
    global_adaptive_cfg = config.get("adaptive_top_k", {}) or {}
    if not isinstance(global_adaptive_cfg, dict):
        raise ValueError("adaptive_top_k must be a mapping when provided.")

    normalized: List[Dict[str, Any]] = []
    seen_names = set()
    for raw in modes:
        if not isinstance(raw, dict):
            raise ValueError("Each inference mode must be a mapping.")
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError("Every inference mode needs a non-empty name.")
        if name in seen_names:
            raise ValueError(f"Duplicate inference mode name: {name}")
        seen_names.add(name)

        adaptive = bool(raw.get("adaptive_top_k", False)) or (
            raw.get("posterior_mass_target") is not None
        )
        raw_cutoff = raw.get("top_k_cutoff")
        if isinstance(raw_cutoff, str) and raw_cutoff.strip().lower() in {"auto", "adaptive"}:
            if not adaptive:
                raise ValueError(
                    f"Mode {name!r} uses top_k_cutoff={raw_cutoff!r} but is not adaptive."
                )
            raw_cutoff = 0.0
        if adaptive and raw_cutoff is None:
            raw_cutoff = 0.0

        cutoff = None if raw_cutoff is None else float(raw_cutoff)
        if cutoff is not None and not 0.0 <= cutoff <= 1.0:
            raise ValueError(f"top_k_cutoff must be in [0, 1], got {cutoff}")

        posterior_mass_target = None
        cutoff_search = None
        if adaptive:
            posterior_mass_target = float(
                raw.get(
                    "posterior_mass_target",
                    global_adaptive_cfg.get("posterior_mass_target", 0.8),
                )
            )
            if not 0.0 < posterior_mass_target <= 1.0:
                raise ValueError(
                    f"posterior_mass_target for adaptive mode {name!r} must be in (0, 1], "
                    f"got {posterior_mass_target}"
                )
            if cutoff is None:
                raise ValueError(
                    f"Adaptive mode {name!r} must compile with a numeric top_k_cutoff seed."
                )
            cutoff_search = _normalize_cutoff_search_config(config, raw)

        normalized.append(
            {
                "name": name,
                "artifact_name": ADAPTIVE_TOP_K_ARTIFACT_LABEL if adaptive else name,
                "top_k_cutoff": cutoff,
                "adaptive_top_k": adaptive,
                "posterior_mass_target": posterior_mass_target,
                "cutoff_search": cutoff_search,
            }
        )
    return normalized


def get_checkpoint_transfer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize checkpoint-transfer configuration."""

    raw = config.get("checkpoint_transfer", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("checkpoint_transfer must be a mapping when provided.")

    configured_mode_names = raw.get("mode_names")
    if configured_mode_names is None:
        mode_names = None
    else:
        if not isinstance(configured_mode_names, list) or not all(str(value).strip() for value in configured_mode_names):
            raise ValueError("checkpoint_transfer.mode_names must be a list of non-empty mode names when provided.")
        mode_names = [str(value).strip() for value in configured_mode_names]

    return {
        "enabled": bool(raw.get("enabled", True)),
        "anchor_mode_name": str(raw.get("anchor_mode_name", "exact")).strip() or "exact",
        "mode_names": mode_names,
        "include_final_checkpoint": bool(raw.get("include_final_checkpoint", True)),
    }


def checkpoint_transfer_run_dir(
    paths: TrainingPaths,
    seed: int,
    n_terms: int,
    mode_name: str,
    anchor_mode_name: str,
) -> Path:
    return paths.runs_root / (
        f"seed_{int(seed)}_terms_{int(n_terms):02d}_transfer_{mode_name}_from_{anchor_mode_name}"
    )


def checkpoint_transfer_checkpoint_path(run_dir_path: Path, segment_index: int) -> Path:
    return run_dir_path / "checkpoints" / f"segment_{int(segment_index):02d}.pt"


def aggregate_checkpoint_path(paths: TrainingPaths, n_terms: int, anchor_mode_name: str) -> Path:
    return paths.root / "aggregate_checkpoints" / (
        f"terms_{int(n_terms):02d}_{anchor_mode_name}_posterior_checkpoints.json"
    )


def mode_artifact_dir(paths: TrainingPaths, n_terms: int, mode: Dict[str, Any]) -> Path:
    artifact_name = str(mode.get("artifact_name", mode["name"]))
    return paths.compiled_root / f"terms_{int(n_terms):02d}" / artifact_name


def compiled_program_path(paths: TrainingPaths, n_terms: int, mode: Dict[str, Any]) -> Path:
    return mode_artifact_dir(paths, n_terms, mode) / "program.py"


def source_program_path(paths: TrainingPaths, n_terms: int) -> Path:
    return paths.program_root / f"sum_terms_{int(n_terms):02d}.spll"


def initial_checkpoint_path(paths: TrainingPaths, seed: int, n_terms: int) -> Path:
    return paths.initial_checkpoints_root / f"seed_{int(seed)}_terms_{int(n_terms):02d}.pt"


def schedule_manifest_path(paths: TrainingPaths, seed: int, n_terms: int) -> Path:
    return paths.schedules_root / f"seed_{int(seed)}_terms_{int(n_terms):02d}_schedule_manifest.json"


def schedule_preview_path(paths: TrainingPaths, seed: int, n_terms: int) -> Path:
    return paths.schedule_previews_root / f"seed_{int(seed)}_terms_{int(n_terms):02d}_preview.jsonl"


def run_dir(paths: TrainingPaths, seed: int, n_terms: int, mode_name: str) -> Path:
    return paths.runs_root / f"seed_{int(seed)}_terms_{int(n_terms):02d}_{mode_name}"


def stable_int_seed(*parts: Any) -> int:
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) % (2**32)
