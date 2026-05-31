from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import numbers
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from mnist_spll_common import (
    CNNClassifier,
    TerminalProgressBar,
    build_train_transform,
    ensure_dir,
    resolve_device,
    resolve_path,
    save_config,
    set_seed,
)
from mnist_spll_pipeline_core import (
    _get_tuple_item,
    compile_spll_program,
    extract_branch_count,
    ADAPTIVE_TOP_K_ARTIFACT_LABEL,
    import_compiled_module,
    make_spll_program,
    threshold_label,
    utc_now_iso,
    write_json,
    load_json,
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


def training_paths(config: Dict[str, Any]) -> TrainingPaths:
    paths_cfg = config.get("paths", {})
    root = ensure_dir(resolve_path(config, paths_cfg.get("output_root", "./outputs/spll_training")))
    generated_root = ensure_dir(root / "generated")
    visualization_root = ensure_dir(root / "visualization")
    return TrainingPaths(
        root=root,
        config_used_path=root / "config_used.yaml",
        data_root=resolve_path(config, paths_cfg.get("data_root", "./data")),
        split_manifest_path=root / "data_split_manifest.json",
        schedules_root=ensure_dir(root / "schedules"),
        schedule_previews_root=ensure_dir(root / "schedules" / "previews"),
        initial_checkpoints_root=ensure_dir(root / "initial_checkpoints"),
        generated_root=generated_root,
        program_root=ensure_dir(generated_root / "spll_programs"),
        compiled_root=ensure_dir(generated_root / "compiled_python"),
        runs_root=ensure_dir(root / "runs"),
        visualization_root=visualization_root,
        tables_root=ensure_dir(visualization_root / "tables"),
        figures_root=ensure_dir(visualization_root / "figures"),
        figures_main_text_root=ensure_dir(visualization_root / "figures" / "main_text"),
        figures_appendix_root=ensure_dir(visualization_root / "figures" / "appendix"),
    )


def get_experiments(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    experiments = config.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Pipeline II config must define a non-empty experiments list.")
    normalized: List[Dict[str, Any]] = []
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
        item["max_steps"] = int(raw.get("max_steps", config.get("training", {}).get("max_steps", 1_000_000)))
        normalized.append(item)
    if not normalized:
        raise ValueError("All Pipeline II experiments are disabled.")
    return normalized


def get_seeds(config: Dict[str, Any]) -> List[int]:
    seeds = config.get("seeds", [int(config.get("seed", 42))])
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list.")
    return [int(seed) for seed in seeds]


def _normalized_float_config(value: Any, *, field_name: str, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc


def _normalized_int_config(value: Any, *, field_name: str, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc


def _normalize_cutoff_search_config(config: Dict[str, Any], raw_mode: Dict[str, Any]) -> Dict[str, Any]:
    global_cfg = config.get("adaptive_top_k", {}) or {}
    mode_cfg = raw_mode.get("cutoff_search", {}) or {}
    if not isinstance(global_cfg, dict):
        raise ValueError("adaptive_top_k must be a mapping when provided.")
    if not isinstance(mode_cfg, dict):
        raise ValueError(f"cutoff_search for mode {raw_mode.get('name')!r} must be a mapping.")

    def get_value(key: str, default: Any) -> Any:
        return mode_cfg.get(key, global_cfg.get(key, default))

    probe_cases = _normalized_int_config(get_value("probe_cases", 20), field_name="adaptive_top_k.probe_cases", default=20)
    max_iterations = _normalized_int_config(get_value("max_iterations", 14), field_name="adaptive_top_k.max_iterations", default=14)
    tolerance = _normalized_float_config(get_value("tolerance", 0.02), field_name="adaptive_top_k.tolerance", default=0.02)
    min_cutoff = _normalized_float_config(get_value("min_cutoff", 0.0), field_name="adaptive_top_k.min_cutoff", default=0.0)
    max_cutoff = _normalized_float_config(get_value("max_cutoff", 1.0), field_name="adaptive_top_k.max_cutoff", default=1.0)

    if probe_cases <= 0:
        raise ValueError("adaptive_top_k.probe_cases must be positive.")
    if max_iterations <= 0:
        raise ValueError("adaptive_top_k.max_iterations must be positive.")
    if tolerance < 0.0:
        raise ValueError("adaptive_top_k.tolerance must be non-negative.")
    if not (0.0 <= min_cutoff <= 1.0 and 0.0 <= max_cutoff <= 1.0 and min_cutoff <= max_cutoff):
        raise ValueError(
            "adaptive_top_k min_cutoff/max_cutoff must satisfy 0 <= min_cutoff <= max_cutoff <= 1."
        )

    return {
        "probe_cases": int(probe_cases),
        "max_iterations": int(max_iterations),
        "tolerance": float(tolerance),
        "min_cutoff": float(min_cutoff),
        "max_cutoff": float(max_cutoff),
    }


def get_inference_modes(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    modes = config.get("inference_modes", [])
    if not isinstance(modes, list) or not modes:
        raise ValueError("Pipeline II config must define a non-empty inference_modes list.")
    global_adaptive_cfg = config.get("adaptive_top_k", {}) or {}
    if not isinstance(global_adaptive_cfg, dict):
        raise ValueError("adaptive_top_k must be a mapping when provided.")

    normalized: List[Dict[str, Any]] = []
    seen = set()
    for raw in modes:
        if not isinstance(raw, dict):
            raise ValueError("Each inference mode must be a mapping.")
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError("Every inference mode needs a non-empty name.")
        if name in seen:
            raise ValueError(f"Duplicate inference mode name: {name}")
        seen.add(name)

        adaptive_requested = bool(raw.get("adaptive_top_k", False)) or raw.get("posterior_mass_target") is not None
        raw_cutoff = raw.get("top_k_cutoff")
        if isinstance(raw_cutoff, str) and raw_cutoff.strip().lower() in {"auto", "adaptive"}:
            if not adaptive_requested:
                raise ValueError(f"Mode {name!r} uses top_k_cutoff={raw_cutoff!r} but is not adaptive.")
            raw_cutoff = 0.0
        if adaptive_requested and raw_cutoff is None:
            # Adaptive modes must be compiled through SPLL's approximate code path,
            # because exact artifacts do not expose the mutable TOP_K_CUTOFF global.
            raw_cutoff = 0.0

        cutoff = raw_cutoff
        if cutoff is not None:
            cutoff = float(cutoff)
            if not (0.0 <= cutoff <= 1.0):
                raise ValueError(f"top_k_cutoff must be in [0, 1], got {cutoff}")

        posterior_mass_target = None
        cutoff_search = None
        if adaptive_requested:
            target_raw = raw.get(
                "posterior_mass_target",
                global_adaptive_cfg.get("posterior_mass_target", 0.8),
            )
            posterior_mass_target = float(target_raw)
            if not (0.0 < posterior_mass_target <= 1.0):
                raise ValueError(
                    f"posterior_mass_target for adaptive mode {name!r} must be in (0, 1], "
                    f"got {posterior_mass_target}"
                )
            if cutoff is None:
                raise ValueError(f"Adaptive mode {name!r} must compile with a numeric top_k_cutoff seed.")
            cutoff_search = _normalize_cutoff_search_config(config, raw)

        normalized.append(
            {
                "name": name,
                "artifact_name": ADAPTIVE_TOP_K_ARTIFACT_LABEL if adaptive_requested else name,
                "top_k_cutoff": cutoff,
                "adaptive_top_k": bool(adaptive_requested),
                "posterior_mass_target": posterior_mass_target,
                "cutoff_search": cutoff_search,
            }
        )
    return normalized


def mode_artifact_dir(paths: TrainingPaths, n_terms: int, mode: Dict[str, Any]) -> Path:
    return paths.compiled_root / f"terms_{int(n_terms):02d}" / str(mode.get("artifact_name", mode["name"]))


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


def load_mnist_train_dataset(config: Dict[str, Any]):
    from torchvision import datasets

    paths = training_paths(config)
    transform = build_train_transform({"training": {"normalize": config.get("data", {}).get("normalize", {"mean": 0.1307, "std": 0.3081})}, "paths": config.get("paths", {}), "_config_dir": config["_config_dir"]})
    return datasets.MNIST(root=str(paths.data_root), train=True, download=True, transform=transform)


def load_mnist_train_labels(config: Dict[str, Any]) -> List[int]:
    from torchvision import datasets

    paths = training_paths(config)
    dataset = datasets.MNIST(root=str(paths.data_root), train=True, download=True, transform=None)
    return [int(label) for label in dataset.targets.tolist()]


def build_balanced_split_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    split_seed = int(data_cfg.get("split_seed", 42))
    train_fraction = float(data_cfg.get("train_fraction", 0.8))
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"data.train_fraction must be in (0, 1), got {train_fraction}")

    labels = load_mnist_train_labels(config)
    by_digit: Dict[int, List[int]] = {digit: [] for digit in range(10)}
    for index, label in enumerate(labels):
        by_digit[int(label)].append(int(index))
    per_digit_count = min(len(indices) for indices in by_digit.values())
    train_count = int(math.floor(per_digit_count * train_fraction))
    if train_count <= 0 or train_count >= per_digit_count:
        raise ValueError("Balanced split would produce an empty train or validation subset.")

    train_indices_by_digit: Dict[str, List[int]] = {}
    validation_indices_by_digit: Dict[str, List[int]] = {}
    for digit, indices in by_digit.items():
        rng = np.random.default_rng(stable_int_seed("split", split_seed, digit))
        chosen = np.array(indices, dtype=np.int64)
        rng.shuffle(chosen)
        chosen = chosen[:per_digit_count]
        train_indices_by_digit[str(digit)] = [int(v) for v in chosen[:train_count].tolist()]
        validation_indices_by_digit[str(digit)] = [int(v) for v in chosen[train_count:].tolist()]

    return {
        "created_at_utc": utc_now_iso(),
        "source": "torchvision.datasets.MNIST(train=True)",
        "split_seed": split_seed,
        "train_fraction": train_fraction,
        "per_digit_count": int(per_digit_count),
        "train_count_per_digit": int(train_count),
        "validation_count_per_digit": int(per_digit_count - train_count),
        "train_indices_by_digit": train_indices_by_digit,
        "validation_indices_by_digit": validation_indices_by_digit,
    }


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _pool_from_manifest(split_manifest: Dict[str, Any], split: str, digit: int) -> List[int]:
    key = "train_indices_by_digit" if split == "train" else "validation_indices_by_digit"
    return [int(v) for v in split_manifest[key][str(int(digit))]]


def generate_sum_case(
    split_manifest: Dict[str, Any],
    *,
    base_seed: int,
    n_terms: int,
    step: int,
    split: str = "train",
) -> Dict[str, Any]:
    if split not in {"train", "validation"}:
        raise ValueError(f"Unsupported schedule split: {split}")
    rng = np.random.default_rng(stable_int_seed("sum-case", base_seed, n_terms, step, split))
    labels = [int(v) for v in rng.integers(0, 10, size=int(n_terms)).tolist()]
    ordered_indices: List[Optional[int]] = [None for _ in labels]
    for digit in sorted(set(labels)):
        positions = [idx for idx, label in enumerate(labels) if label == digit]
        pool = _pool_from_manifest(split_manifest, split, digit)
        if len(pool) < len(positions):
            raise ValueError(
                f"Digit {digit} pool for split={split} has {len(pool)} examples, "
                f"but this case needs {len(positions)} distinct examples."
            )
        chosen = rng.choice(np.array(pool, dtype=np.int64), size=len(positions), replace=False)
        for position, global_index in zip(positions, chosen.tolist()):
            ordered_indices[position] = int(global_index)
    return {
        "step": int(step),
        "n_terms": int(n_terms),
        "global_indices": [int(v) for v in ordered_indices if v is not None],
        "labels": labels,
        "true_sum": int(sum(labels)),
    }


def write_schedule_artifacts(
    config: Dict[str, Any],
    paths: TrainingPaths,
    split_manifest: Dict[str, Any],
) -> None:
    preview_size = int(config.get("schedule", {}).get("preview_size", 20))
    for seed in get_seeds(config):
        for experiment in get_experiments(config):
            n_terms = int(experiment["n_terms"])
            max_steps = int(experiment["max_steps"])
            manifest = {
                "created_at_utc": utc_now_iso(),
                "seed": int(seed),
                "n_terms": n_terms,
                "max_steps": max_steps,
                "sampling": "with_replacement_across_steps",
                "distinct_within_case": True,
                "generator": "random_access_per_step_stable_hash",
                "split_manifest": str(paths.split_manifest_path),
            }
            write_json(schedule_manifest_path(paths, seed, n_terms), manifest)
            rows = (
                generate_sum_case(split_manifest, base_seed=seed, n_terms=n_terms, step=step, split="train")
                for step in range(1, preview_size + 1)
            )
            save_jsonl(schedule_preview_path(paths, seed, n_terms), rows)


def build_model_from_config(config: Dict[str, Any]) -> CNNClassifier:
    model_cfg = config.get("model", {})
    return CNNClassifier(model_cfg)


def write_initial_checkpoints(config: Dict[str, Any], paths: TrainingPaths) -> None:
    for seed in get_seeds(config):
        for experiment in get_experiments(config):
            n_terms = int(experiment["n_terms"])
            set_seed(stable_int_seed("init", seed, n_terms))
            model = build_model_from_config(config)
            payload = {
                "state_dict": model.state_dict(),
                "model_config": dict(config.get("model", {})),
                "seed": int(seed),
                "n_terms": n_terms,
                "created_at_utc": utc_now_iso(),
            }
            target = initial_checkpoint_path(paths, seed, n_terms)
            ensure_dir(target.parent)
            torch.save(payload, target)


def write_spll_sources(config: Dict[str, Any], paths: TrainingPaths) -> None:
    for experiment in get_experiments(config):
        n_terms = int(experiment["n_terms"])
        source_program_path(paths, n_terms).write_text(make_spll_program(n_terms), encoding="utf-8")


def compile_training_artifacts(config: Dict[str, Any], paths: TrainingPaths) -> None:
    repo_root = resolve_path(config, config.get("paths", {}).get("repo_root", "../haskell-dppl-main"))
    compile_cfg = config.get("compile", {})
    timeout_sec = int(compile_cfg.get("timeout_sec", 3600))
    stack_arch = compile_cfg.get("stack_arch", "x86_64")
    if stack_arch is not None:
        stack_arch = str(stack_arch)
    count_branches = bool(compile_cfg.get("count_branches", True))
    force_recompile = bool(compile_cfg.get("force_recompile", True))

    write_spll_sources(config, paths)
    targets: List[Dict[str, Any]] = []
    compiled_once = set()
    total = len(get_experiments(config)) * len(get_inference_modes(config))
    progress = TerminalProgressBar(total, desc="Compile training SPLL", unit="targets", enabled=bool(config.get("show_progress", True)))
    for experiment in get_experiments(config):
        n_terms = int(experiment["n_terms"])
        spll_path = source_program_path(paths, n_terms)
        for mode in get_inference_modes(config):
            out_path = compiled_program_path(paths, n_terms, mode)
            compiled_key = str(out_path.resolve())
            compiled_in_this_stage = compiled_key in compiled_once
            if not compiled_in_this_stage:
                compile_spll_program(
                    repo_root=repo_root,
                    spll_path=spll_path,
                    output_py_path=out_path,
                    cutoff=mode.get("top_k_cutoff"),
                    cutoff_mode="global",
                    force_recompile=force_recompile,
                    timeout_sec=timeout_sec,
                    stack_arch=stack_arch,
                    count_branches=count_branches,
                )
                compiled_once.add(compiled_key)
            manifest = {
                "created_at_utc": utc_now_iso(),
                "n_terms": n_terms,
                "mode_name": mode["name"],
                "artifact_name": mode.get("artifact_name", mode["name"]),
                "top_k_cutoff": mode.get("top_k_cutoff"),
                "adaptive_top_k": bool(mode.get("adaptive_top_k", False)),
                "posterior_mass_target": mode.get("posterior_mass_target"),
                "cutoff_search": mode.get("cutoff_search"),
                "count_branches": count_branches,
                "spll_source": str(spll_path),
                "compiled_python": str(out_path),
                "threshold_label": threshold_label(mode.get("top_k_cutoff")),
                "artifact_threshold_label": mode.get("artifact_name", mode["name"]),
                "compiled_in_this_stage": not compiled_in_this_stage,
            }
            write_json(out_path.parent / "compile_manifest.json", manifest)
            targets.append(manifest)
            progress.update(postfix=f"terms={n_terms}, mode={mode['name']}")
    progress.finish(postfix="done")
    write_json(paths.root / "compile_manifest.json", {"created_at_utc": utc_now_iso(), "targets": targets})


def assert_compiled_artifacts_exist(config: Dict[str, Any], paths: TrainingPaths) -> None:
    missing: List[Path] = []
    for experiment in get_experiments(config):
        for mode in get_inference_modes(config):
            path = compiled_program_path(paths, int(experiment["n_terms"]), mode)
            if not path.exists():
                missing.append(path)
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing[:20])
        raise FileNotFoundError(
            "Missing compiled SPLL training artifacts. Run compile in the Rosetta/x86 environment first.\n"
            "Example:\n"
            "  arch -x86_64 zsh -f\n"
            "  source .venv-spll-x86/bin/activate\n"
            "  ./.venv-spll-x86/bin/python run_spll_training_pipeline.py --config mnist_spll_training_config.yaml compile\n"
            f"Missing artifacts:\n{lines}"
        )


def load_compiled_training_module(path: Path, n_terms: int, mode_name: str):
    module_name = f"spll_training_terms_{int(n_terms):02d}_{mode_name}_{hashlib.sha1(str(path).encode()).hexdigest()[:10]}"
    return import_compiled_module(path, module_name)


def _is_numeric_zero(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Real):
        return float(value) == 0.0
    return False


def _is_zero_tensor(value: torch.Tensor) -> bool:
    if value.numel() != 1:
        return False
    return float(value.detach().cpu().item()) == 0.0


def _pruned_zero_tensor(zero_anchor: Optional[torch.Tensor]) -> torch.Tensor:
    if zero_anchor is not None:
        return zero_anchor * 0.0
    return torch.tensor(0.0, dtype=torch.float32)


def zero_anchor_from_model(model: nn.Module) -> torch.Tensor:
    for param in model.parameters():
        if param.requires_grad:
            return param.sum() * 0.0
    raise RuntimeError("Cannot build a differentiable zero anchor: model has no trainable parameters.")


def _as_probability_tensor(
    return_value: Any,
    *,
    allow_pruned_zero: bool = False,
    zero_anchor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(return_value, torch.Tensor):
        if (
            allow_pruned_zero
            and zero_anchor is not None
            and not return_value.requires_grad
            and _is_zero_tensor(return_value)
        ):
            return _pruned_zero_tensor(zero_anchor)
        return return_value
    if allow_pruned_zero and _is_numeric_zero(return_value):
        return _pruned_zero_tensor(zero_anchor)

    probability = _get_tuple_item(return_value, 0)
    if isinstance(probability, torch.Tensor):
        if (
            allow_pruned_zero
            and zero_anchor is not None
            and not probability.requires_grad
            and _is_zero_tensor(probability)
        ):
            return _pruned_zero_tensor(zero_anchor)
        return probability
    if allow_pruned_zero and _is_numeric_zero(probability):
        return _pruned_zero_tensor(zero_anchor)

    raise TypeError(
        "Generated SPLL artifact returned a non-tensor probability. "
        "Pipeline II requires a differentiable torch.Tensor probability, except for literal zero "
        "returned by a fully pruned true-sum path."
    )


def call_true_sum(
    module: Any,
    true_sum: int,
    indices: Sequence[int],
    *,
    allow_pruned_zero: bool = False,
    zero_anchor: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[int]]:
    sig = inspect.signature(module.main.forward)
    if "acc_prob" in sig.parameters:
        result = module.main.forward(int(true_sum), 1.0, *[int(v) for v in indices])
    else:
        result = module.main.forward(int(true_sum), *[int(v) for v in indices])
    p_true = _as_probability_tensor(result, allow_pruned_zero=allow_pruned_zero, zero_anchor=zero_anchor)
    branch_count = extract_branch_count(result)
    return p_true, branch_count


class TensorLRUCache:
    def __init__(self, dataset: Any, *, device: torch.device, cache_device: str, max_items: int, strategy: str) -> None:
        self.dataset = dataset
        self.device = device
        self.cache_device = cache_device
        self.max_items = max(0, int(max_items))
        self.strategy = str(strategy).lower()
        if self.strategy not in {"none", "lru"}:
            raise ValueError("data.image_cache_strategy currently supports 'none' or 'lru'.")
        if self.cache_device not in {"cpu", "device"}:
            raise ValueError("data.image_cache_device must be 'cpu' or 'device'.")
        self._items: OrderedDict[int, torch.Tensor] = OrderedDict()

    def get(self, index: int) -> torch.Tensor:
        index = int(index)
        if self.strategy == "lru" and index in self._items:
            tensor = self._items.pop(index)
            self._items[index] = tensor
            return tensor.to(self.device) if tensor.device != self.device else tensor

        tensor, _ = self.dataset[index]
        target_device = self.device if self.cache_device == "device" else torch.device("cpu")
        tensor = tensor.to(target_device)
        if self.strategy == "lru" and self.max_items > 0:
            self._items[index] = tensor
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
        return tensor.to(self.device) if tensor.device != self.device else tensor

    def clear(self) -> None:
        self._items.clear()


class DifferentiableReadMNIST:
    def __init__(self, model: nn.Module, tensor_cache: TensorLRUCache, device: torch.device) -> None:
        self.model = model
        self.tensor_cache = tensor_cache
        self.device = device

    def __call__(self, global_index: int) -> torch.Tensor:
        x = self.tensor_cache.get(int(global_index)).unsqueeze(0).to(self.device)
        logits = self.model(x)
        return torch.softmax(logits, dim=-1)[0]


class PrecomputedReadMNIST:
    """readMNist replacement backed by a differentiable per-batch probability lookup.

    The generated SPLL training artifact calls readMNist once per image index while it
    evaluates one sum case.  In batched training we first run the CNN on all unique
    images needed by the whole batch, then let each scalar SPLL call reuse the
    corresponding softmax row from this lookup.  The stored tensors are still part of
    the current autograd graph, so gradients flow back to the CNN even though the SPLL
    calls remain sequential Python calls.
    """

    def __init__(self, probabilities_by_index: Dict[int, torch.Tensor]) -> None:
        self.probabilities_by_index = {int(key): value for key, value in probabilities_by_index.items()}

    def __call__(self, global_index: int) -> torch.Tensor:
        index = int(global_index)
        try:
            return self.probabilities_by_index[index]
        except KeyError as exc:
            raise KeyError(
                f"SPLL requested MNIST index {index}, but it was not precomputed for this batch."
            ) from exc


def make_precomputed_read_mnist(
    *,
    model: nn.Module,
    tensor_cache: TensorLRUCache,
    device: torch.device,
    global_indices: Sequence[int],
) -> PrecomputedReadMNIST:
    """Build a differentiable readMNist lookup for all unique indices in a sum batch."""

    ordered_unique_indices = list(dict.fromkeys(int(v) for v in global_indices))
    if not ordered_unique_indices:
        raise ValueError("Cannot build a precomputed readMNist lookup for an empty batch.")
    xs = torch.stack([tensor_cache.get(index).to(device) for index in ordered_unique_indices], dim=0)
    logits = model(xs)
    probabilities = torch.softmax(logits, dim=-1)
    return PrecomputedReadMNIST(
        {index: probabilities[row_idx] for row_idx, index in enumerate(ordered_unique_indices)}
    )


def digit_accuracy(model: nn.Module, dataset: Any, validation_indices: Sequence[int], device: torch.device, batch_size: int) -> float:
    subset = Subset(dataset, [int(v) for v in validation_indices])
    loader = DataLoader(subset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return float(correct / total) if total else 0.0


def extract_validation_indices(split_manifest: Dict[str, Any]) -> List[int]:
    indices: List[int] = []
    for digit in range(10):
        indices.extend(int(v) for v in split_manifest["validation_indices_by_digit"][str(digit)])
    return indices


def tensor_to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def validate_probability_and_loss(p_true: torch.Tensor, loss: torch.Tensor, *, step: int) -> None:
    p_value = tensor_to_float(p_true)
    loss_value = tensor_to_float(loss)
    if math.isnan(p_value) or math.isinf(p_value):
        raise FloatingPointError(f"Invalid true-sum probability at step {step}: {p_value}")
    if p_value < 0.0:
        raise FloatingPointError(f"Negative true-sum probability at step {step}: {p_value}")
    if math.isnan(loss_value) or math.isinf(loss_value):
        raise FloatingPointError(f"Invalid loss at step {step}: {loss_value}")


def grad_norm(model: nn.Module) -> float:
    total_sq = 0.0
    has_grad = False
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            raise FloatingPointError("Encountered NaN/inf gradient.")
        norm = float(grad.norm().detach().cpu().item())
        total_sq += norm * norm
        has_grad = True
    if not has_grad:
        return 0.0
    return math.sqrt(total_sq)


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    elapsed_seconds: float,
    digit_accuracy_value: float,
    config: Dict[str, Any],
    seed: int,
    n_terms: int,
    mode: Dict[str, Any],
    validation_metric_name: str = "digit_accuracy",
    sum_posterior_accuracy: Optional[float] = None,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": int(step),
            "elapsed_seconds": float(elapsed_seconds),
            "digit_accuracy": float(digit_accuracy_value),
            "validation_metric_name": str(validation_metric_name),
            "sum_posterior_accuracy": (
                float(sum_posterior_accuracy) if sum_posterior_accuracy is not None else None
            ),
            "model_config": dict(config.get("model", {})),
            "seed": int(seed),
            "n_terms": int(n_terms),
            "inference_mode": mode["name"],
            "artifact_name": mode.get("artifact_name", mode["name"]),
            "top_k_cutoff": mode.get("top_k_cutoff"),
            "adaptive_top_k": bool(mode.get("adaptive_top_k", False)),
            "posterior_mass_target": mode.get("posterior_mass_target"),
            "cutoff_search": mode.get("cutoff_search"),
            "config_path": str(config.get("_config_path", "")),
            "created_at_utc": utc_now_iso(),
        },
        path,
    )


def write_failure_report(path: Path, *, error: BaseException, context: Dict[str, Any]) -> None:
    write_json(
        path,
        {
            "created_at_utc": utc_now_iso(),
            "error_type": type(error).__name__,
            "error": str(error),
            "context": context,
        },
    )


def cleanup_torch() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def write_csv_header(path: Path, fieldnames: Sequence[str]):
    ensure_dir(path.parent)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
    writer.writeheader()
    return handle, writer


def recent_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))
