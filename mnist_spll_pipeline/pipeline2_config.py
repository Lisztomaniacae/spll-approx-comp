from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from pipeline_support import ensure_dir, resolve_path

PIPELINE2_READ_MNIST_POLICY = "direct_uncached_model_forward_per_generated_call"


def validate_pipeline2_config(config: Dict[str, Any]) -> None:
    """Fail fast on settings that invalidate the direct-uncached benchmark."""

    # Mode validation also rejects all retired Pipeline-II adaptive-cutoff forms.
    get_inference_modes(config)

    data_cfg = config.get("data", {}) or {}
    obsolete_cache_keys = sorted(
        key
        for key in ("image_cache_device", "image_cache_strategy", "image_cache_max_items")
        if key in data_cfg
    )
    if obsolete_cache_keys:
        raise ValueError(
            "Pipeline II no longer supports application-level MNIST caching. Remove obsolete "
            f"data settings: {', '.join(obsolete_cache_keys)}."
        )

    dropout = float((config.get("model", {}) or {}).get("dropout", 0.0))
    if dropout != 0.0:
        raise ValueError(
            "Pipeline II direct-uncached training requires model.dropout=0.0. Generated SPLL "
            "inference can revisit the same MNIST image multiple times in one query; active "
            "dropout would make readMNist return a different distribution across symbolic "
            "branches and would couple pruning to classifier RNG consumption."
        )


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


def get_inference_modes(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate Pipeline II inference modes.

    Pipeline II intentionally supports only exact inference (``top_k_cutoff: null``)
    and fixed numeric cutoffs. Adaptive cutoff selection belongs to Pipeline I and
    must not be reintroduced here because it changes the experiment definition.
    """

    modes = config.get("inference_modes", [])
    if not isinstance(modes, list) or not modes:
        raise ValueError("Pipeline II config must define a non-empty inference_modes list.")

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

        forbidden_adaptive_fields = {
            key for key in ("adaptive_top_k", "posterior_mass_target", "cutoff_search")
            if key in raw
        }
        raw_cutoff = raw.get("top_k_cutoff")
        if forbidden_adaptive_fields or (
            isinstance(raw_cutoff, str) and raw_cutoff.strip().lower() in {"auto", "adaptive"}
        ):
            raise ValueError(
                f"Pipeline II mode {name!r} contains adaptive-cutoff configuration. "
                "Pipeline II supports only exact inference or fixed numeric top_k_cutoff values; "
                "adaptive top-k remains a Pipeline I feature."
            )

        cutoff = None if raw_cutoff is None else float(raw_cutoff)
        if cutoff is not None and not 0.0 <= cutoff <= 1.0:
            raise ValueError(f"top_k_cutoff must be in [0, 1], got {cutoff}")

        normalized.append(
            {
                "name": name,
                "artifact_name": name,
                "top_k_cutoff": cutoff,
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
