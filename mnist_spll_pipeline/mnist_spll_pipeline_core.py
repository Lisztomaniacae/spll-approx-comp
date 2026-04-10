from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import ConcatDataset

from mnist_spll_common import (
    TerminalProgressBar,
    build_eval_transform,
    default_rng,
    ensure_dir,
    load_checkpoint_model,
    load_config,
    load_full_mnist_raw,
    resolve_device,
    resolve_path,
    save_config,
)


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


@dataclass(frozen=True)
class PipelineContext:
    config: Dict[str, Any]
    paths_cfg: Dict[str, Any]
    inference_cfg: Dict[str, Any]
    show_progress: bool
    paths: PipelinePaths


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def threshold_label(cutoff: Optional[float]) -> str:
    if cutoff is None:
        return "exact"
    return f"cutoff_{str(cutoff).replace('.', 'p')}"


def validate_thresholds(thresholds: Sequence[Optional[float]]) -> None:
    for cutoff in thresholds:
        if cutoff is None:
            continue
        if not (0.0 <= float(cutoff) <= 1.0):
            raise ValueError(
                f"Every topKCutoff value must be between 0 and 1 for this repo. Invalid value: {cutoff}"
            )


def get_thresholds(config: Dict[str, Any]) -> List[Optional[float]]:
    thresholds = list(config["inference"].get("approximation_thresholds", [None, 0.001, 0.01, 0.05]))
    validate_thresholds(thresholds)
    return thresholds


def normalize_cutoff_mode(mode: Any) -> str:
    value = str(mode).strip().lower()
    if value not in {"local", "global"}:
        raise ValueError(f"Unsupported cutoff mode: {mode}")
    return value


def get_cutoff_modes(config: Dict[str, Any]) -> List[str]:
    raw_modes = list(config["inference"].get("cutoff_modes", ["local"]))
    if not raw_modes:
        raise ValueError("inference.cutoff_modes must contain at least one mode.")
    ordered: List[str] = []
    seen = set()
    for mode in raw_modes:
        normalized = normalize_cutoff_mode(mode)
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def compiled_program_path(compiled_root: Path, n_terms: int, cutoff_mode: str, cutoff: Optional[float]) -> Path:
    return compiled_root / normalize_cutoff_mode(cutoff_mode) / f"sum_{int(n_terms):02d}" / threshold_label(cutoff) / "program.py"


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
    experiment_root = ensure_dir(outputs_root / "spll_experiments")
    generated_root = ensure_dir(experiment_root / "generated")
    paths = PipelinePaths(
        outputs_root=outputs_root,
        experiment_root=experiment_root,
        generated_root=generated_root,
        program_root=ensure_dir(generated_root / "spll_programs"),
        compiled_root=ensure_dir(generated_root / "compiled_python"),
        inputs_root=ensure_dir(experiment_root / "inputs"),
        staged_experiments_path=experiment_root / "staged_experiments.json",
        compile_manifest_path=experiment_root / "compile_manifest.json",
        inference_runs_path=experiment_root / "inference_runs.json",
        inference_manifest_path=experiment_root / "inference_manifest.json",
        visualization_root=ensure_dir(experiment_root / "visualization"),
    )
    return PipelineContext(
        config=config,
        paths_cfg=paths_cfg,
        inference_cfg=inference_cfg,
        show_progress=bool(inference_cfg.get("show_progress", True)),
        paths=paths,
    )


def make_spll_program(num_terms: int) -> str:
    if num_terms < 1:
        raise ValueError("num_terms must be >= 1")
    args = [f"x{i}" for i in range(num_terms)]
    exprs = [f"readMNist({arg})" for arg in args]
    expr = exprs[0]
    for part in exprs[1:]:
        expr = f"({expr} ++ {part})"
    return (
        "neural readMNist :: (Symbol -> Int) of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n"
        f"main {' '.join(args)} = {expr}\n"
    )


def write_spll_program(path: Path, num_terms: int) -> None:
    ensure_dir(path.parent)
    path.write_text(make_spll_program(num_terms), encoding="utf-8")


def ensure_programs_for_term_counts(program_root: Path, term_counts: Sequence[int]) -> None:
    for n_terms in term_counts:
        write_spll_program(program_root / f"sum_{int(n_terms):02d}.spll", int(n_terms))


def compile_spll_program(
        repo_root: Path,
        spll_path: Path,
        output_py_path: Path,
        cutoff: Optional[float],
        cutoff_mode: str,
        force_recompile: bool,
        timeout_sec: int,
        stack_arch: Optional[str] = None,
        count_branches: bool = False,
) -> None:
    ensure_dir(output_py_path.parent)
    python_lib_src = repo_root / "pythonLib.py"
    python_lib_dst = output_py_path.parent / "pythonLib.py"

    if output_py_path.exists() and not force_recompile:
        if not python_lib_dst.exists():
            shutil.copy2(python_lib_src, python_lib_dst)
        return

    if shutil.which("stack") is None:
        raise RuntimeError("Could not find 'stack' on PATH. Install Stack before running SPLL compilation.")

    args = ["stack"]
    if stack_arch:
        args += ["--arch", stack_arch]
    args += ["run", "--", "-i", str(spll_path)]
    if count_branches:
        args += ["-c"]
    args += ["--cutoffMode", normalize_cutoff_mode(cutoff_mode)]
    if cutoff is not None:
        args += ["-k", str(cutoff)]
    args += ["compile", "-o", str(output_py_path), "-l", "python"]

    completed = subprocess.run(
        args,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SPLL compilation failed.\n"
            f"Command: {' '.join(args)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    shutil.copy2(python_lib_src, python_lib_dst)


def import_compiled_module(module_path: Path, module_name: str):
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_raw_image(image, destination: Path) -> None:
    ensure_dir(destination.parent)
    image.save(destination)


def _get_tuple_item(value: Any, index: int) -> Any:
    try:
        return value[index]
    except Exception:
        pass
    attr_name = "t1" if index == 0 else "t2"
    if hasattr(value, attr_name):
        return getattr(value, attr_name)
    raise TypeError(f"Value does not expose tuple-like index {index}: {value!r}")



def _to_python_scalar(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            item = value.item()
            if isinstance(item, bool):
                return float(int(item))
            if isinstance(item, (int, float)):
                return float(item)
        except Exception:
            pass
    return None



def extract_probability(return_value: Any) -> float:
    scalar = _to_python_scalar(return_value)
    if scalar is not None:
        return float(scalar)

    try:
        probability = _get_tuple_item(return_value, 0)
    except TypeError as exc:
        raise TypeError(
            f"Could not extract probability from compiled SPLL return value: {return_value!r}"
        ) from exc

    scalar = _to_python_scalar(probability)
    if scalar is not None:
        return float(scalar)

    raise TypeError(f"Could not extract probability from compiled SPLL return value: {return_value!r}")



def extract_branch_count(return_value: Any) -> Optional[int]:
    """
    Compiled SPLL probability calls with branch counting enabled return values shaped like:
        T(probability, T(0.0, branch_count))

    The branch count is therefore the nested second component: result[1][1] / result.t2.t2.
    """
    try:
        metadata = _get_tuple_item(return_value, 1)
        branch_count_value = _get_tuple_item(metadata, 1)
    except TypeError:
        return None

    scalar = _to_python_scalar(branch_count_value)
    if scalar is None:
        raise TypeError(f"Could not extract branch count from compiled SPLL return value: {return_value!r}")

    return int(round(float(scalar)))


def build_read_mnist(model_path: Path, device: torch.device, config_path: Path):
    config = load_config(config_path)
    model = load_checkpoint_model(model_path, config, map_location="cpu")
    model.to(device)
    model.eval()
    transform = build_eval_transform(config)
    from PIL import Image

    @lru_cache(maxsize=None)
    def read_mnist(image_path: str) -> List[float]:
        image = Image.open(image_path).convert("L")
        x = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
        return [float(v) for v in probs]

    return read_mnist


def sample_experiments(
        raw_dataset: ConcatDataset,
        inference_indices: Sequence[int],
        num_experiments: int,
        terms_min: int,
        terms_max: int,
        without_replacement_within_experiment: bool,
        rng,
        inputs_root: Path,
        *,
        show_progress: bool,
) -> List[Dict[str, Any]]:
    experiments: List[Dict[str, Any]] = []
    ensure_dir(inputs_root)

    if terms_max > len(inference_indices):
        raise ValueError(
            f"terms_per_sum_max={terms_max} exceeds the inference subset size {len(inference_indices)}."
        )

    term_counts = rng.integers(low=terms_min, high=terms_max + 1, size=num_experiments)
    staging_bar = TerminalProgressBar(
        num_experiments,
        desc="Staging",
        unit="experiments",
        enabled=show_progress and num_experiments > 0,
    )
    for experiment_id, n_terms in enumerate(term_counts, start=1):
        chosen_positions = rng.choice(
            len(inference_indices),
            size=int(n_terms),
            replace=not without_replacement_within_experiment,
        )
        chosen_global_indices = [int(inference_indices[pos]) for pos in chosen_positions]

        run_input_dir = inputs_root / f"experiment_{experiment_id:04d}"
        ensure_dir(run_input_dir)
        image_paths: List[str] = []
        labels: List[int] = []

        for local_idx, global_index in enumerate(chosen_global_indices):
            image, label = raw_dataset[global_index]
            image_name = f"term_{local_idx:02d}_global_{global_index:05d}_label_{label}.png"
            image_path = run_input_dir / image_name
            save_raw_image(image, image_path)
            image_paths.append(str(image_path.resolve()))
            labels.append(int(label))

        experiments.append(
            {
                "experiment_id": experiment_id,
                "n_terms": int(n_terms),
                "global_indices": chosen_global_indices,
                "image_paths": image_paths,
                "labels": labels,
                "true_sum": int(sum(labels)),
            }
        )
        staging_bar.update(postfix=f"exp={experiment_id:04d}, terms={int(n_terms)}")
    staging_bar.finish(postfix="all staged experiments ready")
    return experiments


def posterior_for_experiment(
        module,
        image_paths: Sequence[str],
        max_sum: int,
        *,
        progress_bar: Optional[TerminalProgressBar] = None,
        progress_prefix: str = "",
) -> Dict[str, List[Optional[float]]]:
    posterior: List[float] = []
    branch_counts: List[Optional[int]] = []
    for candidate in range(max_sum + 1):
        result = module.main.forward(candidate, *image_paths)
        posterior.append(extract_probability(result))
        branch_counts.append(extract_branch_count(result))
        if progress_bar is not None:
            progress_bar.update(postfix=f"{progress_prefix} sum={candidate}")
    return {
        "posterior_raw": posterior,
        "branch_counts_raw": branch_counts,
    }


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_staged_experiments(paths: PipelinePaths) -> List[Dict[str, Any]]:
    if not paths.staged_experiments_path.exists():
        raise FileNotFoundError(
            f"Staged experiments not found at {paths.staged_experiments_path}. Run the 'stage' step first."
        )
    payload = load_json(paths.staged_experiments_path)
    experiments = payload.get("experiments") if isinstance(payload, dict) else payload
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"Expected a non-empty experiment list in {paths.staged_experiments_path}.")
    return experiments


def verify_compiled_artifacts(
        paths: PipelinePaths,
        experiments: List[Dict[str, Any]],
        thresholds: Sequence[Optional[float]],
        cutoff_modes: Sequence[str],
) -> None:
    unique_term_counts = sorted({int(exp["n_terms"]) for exp in experiments})
    for cutoff_mode in cutoff_modes:
        for n_terms in unique_term_counts:
            for cutoff in thresholds:
                compiled_py_path = compiled_program_path(paths.compiled_root, n_terms, cutoff_mode, cutoff)
                if not compiled_py_path.exists():
                    raise FileNotFoundError(
                        f"Compiled SPLL program missing: {compiled_py_path}. Run the 'compile' step first."
                    )


def build_compiled_module_loader(
        paths: PipelinePaths,
        cutoff_modes: Sequence[str],
        thresholds: Sequence[Optional[float]],
        experiments: List[Dict[str, Any]],
        read_mnist,
        *,
        show_progress: bool,
):
    verify_compiled_artifacts(paths, experiments, thresholds, cutoff_modes)

    unique_targets = sorted(
        {(normalize_cutoff_mode(mode), int(exp["n_terms"]), cutoff) for exp in experiments for mode in cutoff_modes for cutoff in thresholds},
        key=lambda item: (item[0], item[1], item[2] is not None, float(item[2] or -1.0)),
    )
    total_targets = len(unique_targets)
    loaded_targets = 0
    compiled_modules: Dict[Tuple[int, str, Optional[float]], Any] = {}
    load_bar = TerminalProgressBar(
        total_targets,
        desc="Load compiled",
        unit="targets",
        enabled=show_progress and total_targets > 0,
    )

    def get_module(n_terms: int, cutoff_mode: str, cutoff: Optional[float]):
        nonlocal loaded_targets
        normalized_mode = normalize_cutoff_mode(cutoff_mode)
        key = (int(n_terms), normalized_mode, cutoff)
        cached = compiled_modules.get(key)
        if cached is not None:
            return cached

        label = threshold_label(cutoff)
        compiled_py_path = compiled_program_path(paths.compiled_root, int(n_terms), normalized_mode, cutoff)
        module_name = f"spll_{normalized_mode}_{int(n_terms)}_{label}_{hashlib.sha1(str(compiled_py_path).encode()).hexdigest()[:10]}"
        module = import_compiled_module(compiled_py_path, module_name)
        setattr(module, "readMNist", read_mnist)
        compiled_modules[key] = module
        loaded_targets += 1
        load_bar.update(postfix=f"cutoff_mode={normalized_mode}, terms={int(n_terms)}, cutoff={label}")
        return module

    def finish_loading() -> None:
        if total_targets > 0:
            load_bar.finish(postfix=f"loaded {len(compiled_modules)}/{total_targets} compiled targets used in this run")

    return get_module, finish_loading


def build_stage_metadata(config: Dict[str, Any], stage_name: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "stage": stage_name,
        "created_at_utc": utc_now_iso(),
        "seed": int(config.get("seed", 42)),
        "config_path": str(config.get("_config_path", "")),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", "unknown"),
    }
    if extra:
        payload.update(extra)
    return payload


def stage_config_snapshot(config: Dict[str, Any], destination: Path) -> None:
    save_config(config, destination)


def load_split_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    split_manifest_path = resolve_path(config, config["paths"]["split_manifest"])
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found at {split_manifest_path}. Run train_mnist.py first.")
    return torch.load(split_manifest_path, map_location="cpu")


def build_experiment_source_bundle(config: Dict[str, Any], paths: PipelinePaths) -> Dict[str, Any]:
    return {
        "config_path": str(config.get("_config_path", "")),
        "paths": paths.to_json_dict(),
        "thresholds": get_thresholds(config),
        "cutoff_modes": get_cutoff_modes(config),
        "term_counts": get_configured_term_counts(config),
    }


def sample_and_save_staged_experiments(config: Dict[str, Any], ctx: PipelineContext) -> List[Dict[str, Any]]:
    split_manifest = load_split_manifest(config)
    inference_indices = list(split_manifest["inference_indices"])
    raw_dataset = load_full_mnist_raw(config)
    num_experiments = int(ctx.inference_cfg.get("num_experiments", 200))
    terms_min, terms_max = get_term_count_bounds(config)

    rng = default_rng(config, offset=100)
    experiments = sample_experiments(
        raw_dataset=raw_dataset,
        inference_indices=inference_indices,
        num_experiments=num_experiments,
        terms_min=terms_min,
        terms_max=terms_max,
        without_replacement_within_experiment=bool(ctx.inference_cfg.get("sample_without_replacement_within_experiment", True)),
        rng=rng,
        inputs_root=ctx.paths.inputs_root,
        show_progress=ctx.show_progress,
    )
    return experiments


