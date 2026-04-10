from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from mnist_spll_common import (
    TerminalProgressBar,
    get_model_variants,
    get_variant_model_output_path,
    load_config,
    resolve_device,
    set_seed,
    stage_message,
)
from mnist_spll_pipeline_core import (
    build_compiled_module_loader,
    build_pipeline_context,
    build_read_mnist,
    build_stage_metadata,
    get_cutoff_modes,
    get_thresholds,
    load_staged_experiments,
    posterior_for_experiment,
    stage_config_snapshot,
    threshold_label,
    utc_now_iso,
    write_json,
)



def run_inference_stage(config: Dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    ctx = build_pipeline_context(config)
    show_inner_progress = bool(ctx.inference_cfg.get("show_inner_progress", True))
    model_variants = get_model_variants(config)

    stage_message(1, 3, "Loading trained model variants and staged experiment bundle")
    experiments = load_staged_experiments(ctx.paths)
    device = resolve_device(ctx.inference_cfg.get("device", "auto"), False)

    stage_message(2, 3, "Verifying compiled SPLL artifacts")
    thresholds = get_thresholds(config)
    cutoff_modes = get_cutoff_modes(config)

    stage_message(3, 3, "Running posterior inference for every model variant, cutoff mode, and threshold")
    raw_runs: List[Dict[str, Any]] = []
    inference_total = len(model_variants) * len(experiments) * len(cutoff_modes) * len(thresholds)
    inference_bar = TerminalProgressBar(
        inference_total,
        desc="Inference",
        unit="runs",
        enabled=ctx.show_progress and inference_total > 0,
    )

    for variant in model_variants:
        model_id = variant["id"]
        target_accuracy = float(variant["target_accuracy"])
        model_path = get_variant_model_output_path(config, model_id)
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model variant '{model_id}' not found at {model_path}. Run train first.")
        checkpoint_meta = torch.load(model_path, map_location="cpu")
        read_mnist = build_read_mnist(model_path, device, Path(config["_config_path"]))
        get_compiled_module, finish_loading = build_compiled_module_loader(
            ctx.paths,
            cutoff_modes,
            thresholds,
            experiments,
            read_mnist,
            show_progress=ctx.show_progress,
        )

        for cutoff_mode in cutoff_modes:
            for experiment in experiments:
                n_terms = int(experiment["n_terms"])
                image_paths = list(experiment["image_paths"])
                max_sum = 9 * n_terms
                for cutoff in thresholds:
                    label = threshold_label(cutoff)
                    module = get_compiled_module(n_terms, cutoff_mode, cutoff)
                    per_run_bar = TerminalProgressBar(
                        max_sum + 1,
                        desc="  Posterior",
                        unit="sums",
                        enabled=ctx.show_progress and show_inner_progress and (max_sum + 1) > 0,
                    )
                    started_at = utc_now_iso()
                    started = time.perf_counter()
                    posterior_trace = posterior_for_experiment(
                        module,
                        image_paths,
                        max_sum=max_sum,
                        progress_bar=per_run_bar,
                        progress_prefix=(
                            f"model={model_id}, cutoff_mode={cutoff_mode}, exp={experiment['experiment_id']:04d}, "
                            f"terms={n_terms}, cutoff={label},"
                        ),
                    )
                    runtime_sec = time.perf_counter() - started
                    finished_at = utc_now_iso()
                    per_run_bar.finish(
                        postfix=(
                            f"model={model_id}, cutoff_mode={cutoff_mode}, exp={experiment['experiment_id']:04d}, "
                            f"terms={n_terms}, cutoff={label}, runtime={runtime_sec:.2f}s"
                        )
                    )
                    raw_runs.append(
                        {
                            "model_id": model_id,
                            "target_accuracy": target_accuracy,
                            "selected_epoch": int(checkpoint_meta.get("selected_epoch", checkpoint_meta.get("best_epoch", -1))),
                            "selected_test_accuracy": float(
                                checkpoint_meta.get(
                                    "selected_test_accuracy",
                                    checkpoint_meta.get("best_test_accuracy", 0.0),
                                )
                            ),
                            "experiment_id": int(experiment["experiment_id"]),
                            "cutoff_mode": cutoff_mode,
                            "n_terms": n_terms,
                            "cutoff": cutoff,
                            "threshold_label": label,
                            "candidate_sums": list(range(max_sum + 1)),
                            "posterior_raw": [float(value) for value in posterior_trace["posterior_raw"]],
                            "branch_counts_raw": [
                                None if value is None else int(value) for value in posterior_trace["branch_counts_raw"]
                            ],
                            "runtime_sec": float(runtime_sec),
                            "started_at_utc": started_at,
                            "finished_at_utc": finished_at,
                            "true_sum": int(experiment["true_sum"]),
                            "labels": [int(v) for v in experiment["labels"]],
                            "global_indices": [int(v) for v in experiment["global_indices"]],
                            "image_paths": image_paths,
                            "model_path": str(model_path),
                            "compiled_program_path": str(
                                ctx.paths.compiled_root / cutoff_mode / f"sum_{n_terms:02d}" / label / "program.py"
                            ),
                        }
                    )
                    inference_bar.update(
                        postfix=(
                            f"model={model_id}, cutoff_mode={cutoff_mode}, exp={experiment['experiment_id']:04d}, "
                            f"terms={n_terms}, cutoff={label}, runtime={runtime_sec:.2f}s"
                        )
                    )
        finish_loading()

    inference_bar.finish(postfix="all inference runs complete")

    write_json(
        ctx.paths.inference_manifest_path,
        {
            "metadata": build_stage_metadata(
                config,
                "infer",
                extra={
                    "model_variants": [variant["id"] for variant in model_variants],
                    "device": str(device),
                    "cutoff_modes": cutoff_modes,
                    "thresholds": thresholds,
                    "num_runs": len(raw_runs),
                    "show_inner_progress": show_inner_progress,
                    "count_branches": bool(ctx.inference_cfg.get("count_branches", True)),
                    "paths": ctx.paths.to_json_dict(),
                },
            ),
            "experiments_source": str(ctx.paths.staged_experiments_path),
            "compile_manifest_source": str(ctx.paths.compile_manifest_path),
        },
    )
    write_json(
        ctx.paths.inference_runs_path,
        {
            "metadata": build_stage_metadata(
                config,
                "infer_runs",
                extra={
                    "model_variants": [variant["id"] for variant in model_variants],
                    "device": str(device),
                    "cutoff_modes": cutoff_modes,
                    "thresholds": thresholds,
                    "num_runs": len(raw_runs),
                    "count_branches": bool(ctx.inference_cfg.get("count_branches", True)),
                    "paths": ctx.paths.to_json_dict(),
                },
            ),
            "runs": raw_runs,
        },
    )
    stage_config_snapshot(config, ctx.paths.experiment_root / "infer_config_used.yaml")
    print(f"Saved inference manifest to: {ctx.paths.inference_manifest_path}")
    print(f"Saved raw inference runs to: {ctx.paths.inference_runs_path}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPLL posterior inference for staged MNIST addition experiments.")
    parser.add_argument("--config", required=True, help="Path to the shared YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_inference_stage(config)


if __name__ == "__main__":
    main()
