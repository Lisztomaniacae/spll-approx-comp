from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from inference_engine import (
    InferenceRunEngine,
    ModelInferenceContext,
    normalize_read_mnist_cache_policy,
    warm_up_read_mnist,
)
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
    stage_config_snapshot,
    threshold_spec_for_json,
    write_json,
)



def run_inference_stage(config: Dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    ctx = build_pipeline_context(config)
    show_inner_progress = bool(ctx.inference_cfg.get("show_inner_progress", True))
    read_mnist_cache_policy = normalize_read_mnist_cache_policy(
        ctx.inference_cfg.get("read_mnist_cache_policy", "uncached")
    )
    read_mnist_warmup_calls = int(ctx.inference_cfg.get("read_mnist_warmup_calls", 0))
    if read_mnist_warmup_calls < 0:
        raise ValueError("inference.read_mnist_warmup_calls must be >= 0")
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

    model_warmup_stats: Dict[str, Dict[str, Any]] = {}

    for variant in model_variants:
        model_id = variant["id"]
        target_accuracy = float(variant["target_accuracy"])
        model_path = get_variant_model_output_path(config, model_id)
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model variant '{model_id}' not found at {model_path}. Run train first.")
        checkpoint_meta = torch.load(model_path, map_location="cpu")
        model_context = ModelInferenceContext.from_checkpoint(
            model_id=model_id,
            visualization_group=str(variant.get("visualization_group", "main")),
            target_accuracy=target_accuracy,
            model_path=model_path,
            checkpoint_meta=checkpoint_meta,
        )
        read_mnist = build_read_mnist(model_path, device, Path(config["_config_path"]))
        warmup_stats = warm_up_read_mnist(read_mnist, experiments, read_mnist_warmup_calls)
        model_warmup_stats[model_id] = warmup_stats
        get_compiled_module, finish_loading = build_compiled_module_loader(
            ctx.paths,
            cutoff_modes,
            thresholds,
            experiments,
            read_mnist,
            show_progress=ctx.show_progress,
        )
        engine = InferenceRunEngine(
            paths=ctx.paths,
            model=model_context,
            get_compiled_module=get_compiled_module,
            show_progress=ctx.show_progress,
            show_inner_progress=show_inner_progress,
            progress_bar=inference_bar,
            read_mnist_cache_policy=read_mnist_cache_policy,
            read_mnist_warmup_stats=warmup_stats,
        )
        raw_runs.extend(
            engine.run_many(
                experiments=experiments,
                cutoff_modes=cutoff_modes,
                thresholds=thresholds,
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
                    "model_visualization_groups": {
                        variant["id"]: variant.get("visualization_group", "main") for variant in model_variants
                    },
                    "device": str(device),
                    "cutoff_modes": cutoff_modes,
                    "thresholds": [threshold_spec_for_json(threshold) for threshold in thresholds],
                    "num_runs": len(raw_runs),
                    "show_inner_progress": show_inner_progress,
                    "count_branches": bool(ctx.inference_cfg.get("count_branches", True)),
                    "true_candidate_trace": True,
                    "read_mnist_cache_policy": read_mnist_cache_policy,
                    "read_mnist_warmup_calls": read_mnist_warmup_calls,
                    "read_mnist_warmup_by_model": model_warmup_stats,
                    "threshold_order_policy": "rotated_by_experiment_index",
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
                    "model_visualization_groups": {
                        variant["id"]: variant.get("visualization_group", "main") for variant in model_variants
                    },
                    "device": str(device),
                    "cutoff_modes": cutoff_modes,
                    "thresholds": [threshold_spec_for_json(threshold) for threshold in thresholds],
                    "num_runs": len(raw_runs),
                    "count_branches": bool(ctx.inference_cfg.get("count_branches", True)),
                    "true_candidate_trace": True,
                    "read_mnist_cache_policy": read_mnist_cache_policy,
                    "read_mnist_warmup_calls": read_mnist_warmup_calls,
                    "read_mnist_warmup_by_model": model_warmup_stats,
                    "threshold_order_policy": "rotated_by_experiment_index",
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
