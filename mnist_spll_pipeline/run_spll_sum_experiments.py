from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
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
    set_seed,
    stage_message,
)


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


def compile_spll_program(
    repo_root: Path,
    spll_path: Path,
    output_py_path: Path,
    cutoff: Optional[float],
    force_recompile: bool,
    timeout_sec: int,
    stack_arch: Optional[str] = None,
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


def extract_probability(return_value: Any) -> float:
    if isinstance(return_value, (int, float)):
        return float(return_value)
    try:
        first = return_value[0]
        if isinstance(first, (int, float)):
            return float(first)
    except Exception:
        pass
    if hasattr(return_value, "t1") and isinstance(return_value.t1, (int, float)):
        return float(return_value.t1)
    raise TypeError(f"Could not extract probability from compiled SPLL return value: {return_value!r}")


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
) -> List[float]:
    posterior: List[float] = []
    for candidate in range(max_sum + 1):
        result = module.main.forward(candidate, *image_paths)
        posterior.append(extract_probability(result))
        if progress_bar is not None:
            progress_bar.update(postfix=f"{progress_prefix} sum={candidate}")
    return posterior


def top_predictions(posterior: Sequence[float], k: int) -> List[Dict[str, float]]:
    indexed = sorted(enumerate(posterior), key=lambda item: item[1], reverse=True)[:k]
    return [{"sum": int(idx), "probability": float(prob)} for idx, prob in indexed]


def normalize_distribution(values: Sequence[float]) -> List[float]:
    total = float(sum(values))
    if total <= 0:
        return [0.0 for _ in values]
    return [float(v) / total for v in values]


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_plots(summary_rows: List[Dict[str, Any]], detailed_rows: List[Dict[str, Any]], term_counts: List[int], plots_dir: Path) -> None:
    ensure_dir(plots_dir)

    plt.figure(figsize=(8, 4.5))
    plt.hist(term_counts, bins=range(min(term_counts), max(term_counts) + 2), align="left", rwidth=0.85)
    plt.xlabel("Number of digits summed")
    plt.ylabel("Experiment count")
    plt.title("Distribution of sampled term counts")
    plt.tight_layout()
    plt.savefig(plots_dir / "term_count_histogram.png", dpi=180)
    plt.close()

    labels = [row["threshold_label"] for row in summary_rows]
    runtimes = [row["mean_runtime_sec"] for row in summary_rows]
    accuracies = [row["accuracy"] for row in summary_rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(labels, runtimes, marker="o")
    plt.xlabel("Inference setting")
    plt.ylabel("Mean runtime (s)")
    plt.title("Mean runtime by SPLL cutoff")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "runtime_vs_cutoff.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(labels, accuracies, marker="o")
    plt.xlabel("Inference setting")
    plt.ylabel("Accuracy")
    plt.title("Prediction accuracy by SPLL cutoff")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_vs_cutoff.png", dpi=180)
    plt.close()


def summarize_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["threshold_label"]].append(row)

    summary: List[Dict[str, Any]] = []
    for label, items in grouped.items():
        summary.append(
            {
                "threshold_label": label,
                "cutoff": items[0]["cutoff"],
                "experiments": len(items),
                "accuracy": sum(int(item["correct"]) for item in items) / len(items),
                "mean_runtime_sec": mean(item["runtime_sec"] for item in items),
                "median_runtime_sec": median(item["runtime_sec"] for item in items),
                "mean_confidence": mean(item["confidence"] for item in items),
                "mean_posterior_mass": mean(item["posterior_mass"] for item in items),
            }
        )
    summary.sort(key=lambda row: (row["cutoff"] is not None, float(row["cutoff"] or -1.0)))
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_pipeline_context(config: Dict[str, Any]) -> Dict[str, Any]:
    paths_cfg = config["paths"]
    inference_cfg = config["inference"]
    outputs_root = resolve_path(config, paths_cfg.get("outputs_root", "./outputs"))
    experiment_root = ensure_dir(outputs_root / "spll_experiments")
    generated_root = ensure_dir(experiment_root / "generated")
    return {
        "paths_cfg": paths_cfg,
        "inference_cfg": inference_cfg,
        "show_progress": bool(inference_cfg.get("show_progress", True)),
        "outputs_root": outputs_root,
        "experiment_root": experiment_root,
        "generated_root": generated_root,
        "program_root": ensure_dir(generated_root / "spll_programs"),
        "compiled_root": ensure_dir(generated_root / "compiled_python"),
        "sampled_experiments_path": experiment_root / "sampled_experiments.json",
    }


def stage_experiments_only(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    ctx = load_pipeline_context(config)
    paths_cfg = ctx["paths_cfg"]
    inference_cfg = ctx["inference_cfg"]

    stage_message(1, 2, "Loading split manifest and MNIST metadata")
    split_manifest_path = resolve_path(config, paths_cfg["split_manifest"])
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found at {split_manifest_path}. Run train_mnist.py first.")

    split_manifest = torch.load(split_manifest_path, map_location="cpu")
    inference_indices = list(split_manifest["inference_indices"])
    raw_dataset = load_full_mnist_raw(config)

    num_experiments = int(inference_cfg.get("num_experiments", 200))
    terms_min = int(inference_cfg.get("terms_per_sum_min", 1))
    terms_max = int(inference_cfg.get("terms_per_sum_max", 4))
    if terms_min < 1 or terms_max < terms_min:
        raise ValueError(f"Invalid term count bounds: min={terms_min}, max={terms_max}")

    stage_message(2, 2, "Sampling and writing staged MNIST addition experiments")
    rng = default_rng(config, offset=100)
    inputs_root = ensure_dir(ctx["experiment_root"] / "inputs")
    experiments = sample_experiments(
        raw_dataset=raw_dataset,
        inference_indices=inference_indices,
        num_experiments=num_experiments,
        terms_min=terms_min,
        terms_max=terms_max,
        without_replacement_within_experiment=bool(inference_cfg.get("sample_without_replacement_within_experiment", True)),
        rng=rng,
        inputs_root=inputs_root,
        show_progress=ctx["show_progress"],
    )
    write_json(ctx["sampled_experiments_path"], experiments)
    save_config(config, ctx["experiment_root"] / "config_used.yaml")

    unique_term_counts = sorted({int(exp["n_terms"]) for exp in experiments})
    for n_terms in unique_term_counts:
        write_spll_program(ctx["program_root"] / f"sum_{n_terms:02d}.spll", n_terms)

    print(f"Saved staged experiments to: {ctx['sampled_experiments_path']}")
    return experiments


def load_staged_experiments(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    sampled_experiments_path = ctx["sampled_experiments_path"]
    if not sampled_experiments_path.exists():
        raise FileNotFoundError(
            f"Staged experiments not found at {sampled_experiments_path}. Run the 'stage' command first."
        )
    experiments = load_json(sampled_experiments_path)
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"Expected a non-empty experiment list in {sampled_experiments_path}.")
    return experiments


def compile_only(config: Dict[str, Any]) -> None:
    ctx = load_pipeline_context(config)
    inference_cfg = ctx["inference_cfg"]
    paths_cfg = ctx["paths_cfg"]

    stage_message(1, 2, "Loading staged experiments and compiler settings")
    experiments = load_staged_experiments(ctx)
    repo_root = resolve_path(config, paths_cfg["repo_root"])
    if not repo_root.exists():
        raise FileNotFoundError(f"Configured repo_root does not exist: {repo_root}")

    thresholds = list(inference_cfg.get("approximation_thresholds", [None, 0.001, 0.01, 0.05]))
    validate_thresholds(thresholds)
    unique_term_counts = sorted({int(exp["n_terms"]) for exp in experiments})
    force_recompile = bool(inference_cfg.get("force_recompile", False))
    timeout_sec = int(inference_cfg.get("compile_timeout_sec", 600))
    stack_arch = inference_cfg.get("stack_arch")
    if stack_arch is not None:
        stack_arch = str(stack_arch)

    stage_message(2, 2, "Compiling staged SPLL programs")
    compile_total = len(unique_term_counts) * len(thresholds)
    compile_bar = TerminalProgressBar(
        compile_total,
        desc="Compile",
        unit="targets",
        enabled=ctx["show_progress"] and compile_total > 0,
    )
    for n_terms in unique_term_counts:
        spll_path = ctx["program_root"] / f"sum_{n_terms:02d}.spll"
        if not spll_path.exists():
            write_spll_program(spll_path, n_terms)
        for cutoff in thresholds:
            label = threshold_label(cutoff)
            compiled_py_path = ctx["compiled_root"] / f"sum_{n_terms:02d}" / label / "program.py"
            compile_spll_program(
                repo_root=repo_root,
                spll_path=spll_path,
                output_py_path=compiled_py_path,
                cutoff=cutoff,
                force_recompile=force_recompile,
                timeout_sec=timeout_sec,
                stack_arch=stack_arch,
            )
            compile_bar.update(postfix=f"terms={n_terms}, mode={label}")
    compile_bar.finish(postfix="all compilation targets ready")
    print(f"Saved compiled artifacts under: {ctx['compiled_root']}")


def verify_compiled_artifacts(ctx: Dict[str, Any], experiments: List[Dict[str, Any]], thresholds: Sequence[Optional[float]]) -> None:
    unique_term_counts = sorted({int(exp["n_terms"]) for exp in experiments})
    for n_terms in unique_term_counts:
        for cutoff in thresholds:
            label = threshold_label(cutoff)
            compiled_py_path = ctx["compiled_root"] / f"sum_{n_terms:02d}" / label / "program.py"
            if not compiled_py_path.exists():
                raise FileNotFoundError(
                    f"Compiled SPLL program missing: {compiled_py_path}. Run the 'compile' command first."
                )


def build_compiled_module_loader(
    config: Dict[str, Any],
    experiments: List[Dict[str, Any]],
    read_mnist,
):
    ctx = load_pipeline_context(config)
    thresholds = list(ctx["inference_cfg"].get("approximation_thresholds", [None, 0.001, 0.01, 0.05]))
    validate_thresholds(thresholds)
    verify_compiled_artifacts(ctx, experiments, thresholds)

    unique_targets = sorted(
        {(int(exp["n_terms"]), cutoff) for exp in experiments for cutoff in thresholds},
        key=lambda item: (item[0], item[1] is not None, float(item[1] or -1.0)),
    )
    total_targets = len(unique_targets)
    loaded_targets = 0
    compiled_modules: Dict[Tuple[int, Optional[float]], Any] = {}
    load_bar = TerminalProgressBar(
        total_targets,
        desc="Load compiled",
        unit="targets",
        enabled=ctx["show_progress"] and total_targets > 0,
    )

    def get_module(n_terms: int, cutoff: Optional[float]):
        nonlocal loaded_targets
        key = (int(n_terms), cutoff)
        cached = compiled_modules.get(key)
        if cached is not None:
            return cached

        label = threshold_label(cutoff)
        compiled_py_path = ctx["compiled_root"] / f"sum_{int(n_terms):02d}" / label / "program.py"
        if not compiled_py_path.exists():
            raise FileNotFoundError(
                f"Compiled SPLL program missing: {compiled_py_path}. Run the 'compile' command first."
            )
        module_name = f"spll_{int(n_terms)}_{label}_{hashlib.sha1(str(compiled_py_path).encode()).hexdigest()[:10]}"
        module = import_compiled_module(compiled_py_path, module_name)
        setattr(module, "readMNist", read_mnist)
        compiled_modules[key] = module
        loaded_targets += 1
        load_bar.update(postfix=f"terms={int(n_terms)}, mode={label}")
        return module

    def finish_loading() -> None:
        used_targets = len(compiled_modules)
        if total_targets > 0:
            load_bar.finish(postfix=f"loaded {used_targets}/{total_targets} compiled targets used in this run")

    return get_module, finish_loading


def infer_only(config: Dict[str, Any]) -> None:
    ctx = load_pipeline_context(config)
    inference_cfg = ctx["inference_cfg"]
    paths_cfg = ctx["paths_cfg"]
    stage_count = 4 if bool(inference_cfg.get("visualize", True)) else 3
    show_inner_progress = bool(inference_cfg.get("show_inner_progress", True))

    stage_message(1, stage_count, "Loading trained model and staged experiments")
    model_path = resolve_path(config, paths_cfg["model_output"])
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train_mnist.py first.")
    experiments = load_staged_experiments(ctx)

    device = resolve_device(inference_cfg.get("device", "auto"), False)
    read_mnist = build_read_mnist(model_path, device, Path(config["_config_path"]))

    stage_message(2, stage_count, "Verifying compiled SPLL Python artifacts")
    thresholds = list(inference_cfg.get("approximation_thresholds", [None, 0.001, 0.01, 0.05]))
    validate_thresholds(thresholds)
    get_compiled_module, finish_loading = build_compiled_module_loader(config, experiments, read_mnist)

    validate_thresholds(thresholds)
    top_n = int(inference_cfg.get("top_predictions_to_store", 5))
    detailed_rows: List[Dict[str, Any]] = []
    inference_total = len(experiments) * len(thresholds)

    stage_message(3, stage_count, "Running posterior inference over staged experiments")
    inference_bar = TerminalProgressBar(
        inference_total,
        desc="Inference",
        unit="runs",
        enabled=ctx["show_progress"] and inference_total > 0,
    )
    for experiment in experiments:
        n_terms = int(experiment["n_terms"])
        image_paths = list(experiment["image_paths"])
        max_sum = 9 * n_terms
        for cutoff in thresholds:
            label = threshold_label(cutoff)
            module = get_compiled_module(n_terms, cutoff)
            per_run_bar = TerminalProgressBar(
                max_sum + 1,
                desc="  Posterior",
                unit="sums",
                enabled=ctx["show_progress"] and show_inner_progress and (max_sum + 1) > 0,
            )
            started = time.perf_counter()
            posterior_raw = posterior_for_experiment(
                module,
                image_paths,
                max_sum=max_sum,
                progress_bar=per_run_bar,
                progress_prefix=(
                    f"exp={experiment['experiment_id']:04d}, terms={n_terms}, mode={label},"
                ),
            )
            runtime_sec = time.perf_counter() - started
            per_run_bar.finish(
                postfix=(
                    f"exp={experiment['experiment_id']:04d}, terms={n_terms}, mode={label}, "
                    f"runtime={runtime_sec:.2f}s"
                )
            )
            posterior = normalize_distribution(posterior_raw)
            predicted_sum = int(max(range(len(posterior)), key=lambda idx: posterior[idx]))
            confidence = float(posterior[predicted_sum]) if posterior else 0.0
            top_preds = top_predictions(posterior, top_n)
            row = {
                "experiment_id": experiment["experiment_id"],
                "threshold_label": label,
                "cutoff": cutoff,
                "n_terms": n_terms,
                "true_sum": experiment["true_sum"],
                "predicted_sum": predicted_sum,
                "correct": int(predicted_sum == experiment["true_sum"]),
                "runtime_sec": runtime_sec,
                "confidence": confidence,
                "posterior_mass": float(sum(posterior_raw)),
                "labels": json.dumps(experiment["labels"]),
                "image_paths": json.dumps(image_paths),
                "top_predictions": json.dumps(top_preds),
            }
            detailed_rows.append(row)
            inference_bar.update(
                postfix=(
                    f"exp={experiment['experiment_id']:04d}, "
                    f"terms={n_terms}, mode={label}, runtime={runtime_sec:.2f}s"
                )
            )
    inference_bar.finish(postfix="all inference runs complete")
    finish_loading()

    stage_message(4 if stage_count == 4 else 3, stage_count, "Summarizing and writing result tables")
    summary_rows = summarize_results(detailed_rows)
    write_csv(ctx["experiment_root"] / "detailed_results.csv", detailed_rows)
    write_csv(ctx["experiment_root"] / "summary_results.csv", summary_rows)
    write_json(ctx["experiment_root"] / "summary_results.json", summary_rows)

    if bool(inference_cfg.get("visualize", True)):
        stage_message(4, 4, "Rendering plots")
        save_plots(
            summary_rows=summary_rows,
            detailed_rows=detailed_rows,
            term_counts=[int(exp["n_terms"]) for exp in experiments],
            plots_dir=ctx["experiment_root"] / "plots",
        )

    print(f"Saved experiment bundle to: {ctx['experiment_root']}")
    for row in summary_rows:
        print(
            f"{row['threshold_label']}: accuracy={row['accuracy']:.4%}, "
            f"mean_runtime={row['mean_runtime_sec']:.4f}s, mean_confidence={row['mean_confidence']:.4f}"
        )


def run_all(config: Dict[str, Any]) -> None:
    stage_experiments_only(config)
    compile_only(config)
    infer_only(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage, compile, and run MNIST sum experiments with SPLL exact/approximate inference.")
    parser.add_argument("--config", required=True, help="Path to the shared YAML config.")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "stage", "compile", "infer"],
        help="Which part of the pipeline to run. Default: all",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    if args.command == "stage":
        stage_experiments_only(config)
    elif args.command == "compile":
        compile_only(config)
    elif args.command == "infer":
        infer_only(config)
    else:
        run_all(config)


if __name__ == "__main__":
    main()
