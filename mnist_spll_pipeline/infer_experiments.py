from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch

from inference_engine import (
    InferenceRunEngine,
    ModelInferenceContext,
    normalize_read_mnist_cache_policy,
    warm_up_read_mnist,
)
from mnist_model import resolve_device, set_seed
from pipeline1_config import (
    GLOBAL_CUTOFF_MODE,
    build_pipeline_context,
    get_thresholds,
    threshold_spec_for_json,
)
from pipeline1_data import build_read_mnist, load_staged_experiments
from pipeline1_models import get_model_variants, get_variant_model_output_path
from pipeline_support import (
    TerminalProgressBar,
    build_stage_metadata,
    run_configured_stage_cli,
    save_config,
    stage_message,
    write_json,
)
from spll_artifacts import build_compiled_module_loader


def run_inference_stage(config: Dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    context = build_pipeline_context(config)
    context.paths.ensure_inference_dirs()
    inference_cfg = context.inference_cfg

    show_inner_progress = bool(inference_cfg.get("show_inner_progress", True))
    cache_policy = normalize_read_mnist_cache_policy(
        inference_cfg.get("read_mnist_cache_policy", "uncached")
    )
    warmup_calls = int(inference_cfg.get("read_mnist_warmup_calls", 0))
    if warmup_calls < 0:
        raise ValueError("inference.read_mnist_warmup_calls must be >= 0")

    model_variants = get_model_variants(config)
    stage_message(1, 3, "Loading trained model variants and staged experiment bundle")
    experiments = load_staged_experiments(context.paths)
    device = resolve_device(inference_cfg.get("device", "auto"), False)

    stage_message(2, 3, "Verifying compiled SPLL artifacts")
    thresholds = get_thresholds(config)
    cutoff_modes = [GLOBAL_CUTOFF_MODE]

    stage_message(3, 3, "Running posterior inference for every model variant and threshold")
    raw_runs: List[Dict[str, Any]] = []
    inference_total = len(model_variants) * len(experiments) * len(thresholds)
    progress = TerminalProgressBar(
        inference_total,
        desc="Inference",
        unit="runs",
        enabled=context.show_progress and inference_total > 0,
    )
    warmup_by_model: Dict[str, Dict[str, Any]] = {}

    for variant in model_variants:
        model_id = variant["id"]
        model_path = get_variant_model_output_path(config, model_id)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model variant '{model_id}' not found at {model_path}. Run train first."
            )

        checkpoint_meta = torch.load(model_path, map_location="cpu")
        model_context = ModelInferenceContext.from_checkpoint(
            model_id=model_id,
            visualization_group=str(variant.get("visualization_group", "main")),
            target_accuracy=float(variant["target_accuracy"]),
            model_path=model_path,
            checkpoint_meta=checkpoint_meta,
        )
        read_mnist = build_read_mnist(model_path, device, Path(config["_config_path"]))
        warmup_stats = warm_up_read_mnist(read_mnist, experiments, warmup_calls)
        warmup_by_model[model_id] = warmup_stats
        get_compiled_module, finish_loading = build_compiled_module_loader(
            context.paths,
            cutoff_modes,
            thresholds,
            experiments,
            read_mnist,
            show_progress=context.show_progress,
        )
        try:
            engine = InferenceRunEngine(
                paths=context.paths,
                model=model_context,
                get_compiled_module=get_compiled_module,
                show_progress=context.show_progress,
                show_inner_progress=show_inner_progress,
                progress_bar=progress,
                read_mnist_cache_policy=cache_policy,
                read_mnist_warmup_stats=warmup_stats,
            )
            raw_runs.extend(
                engine.run_many(
                    experiments=experiments,
                    cutoff_modes=cutoff_modes,
                    thresholds=thresholds,
                )
            )
        finally:
            finish_loading()

    progress.finish(postfix="all inference runs complete")

    metadata_extra = {
        "model_variants": [variant["id"] for variant in model_variants],
        "model_visualization_groups": {
            variant["id"]: variant.get("visualization_group", "main")
            for variant in model_variants
        },
        "device": str(device),
        "cutoff_modes": cutoff_modes,
        "thresholds": [threshold_spec_for_json(item) for item in thresholds],
        "num_runs": len(raw_runs),
        "count_branches": bool(inference_cfg.get("count_branches", True)),
        "true_candidate_trace": True,
        "read_mnist_cache_policy": cache_policy,
        "read_mnist_warmup_calls": warmup_calls,
        "read_mnist_warmup_by_model": warmup_by_model,
        "threshold_order_policy": "rotated_by_experiment_index",
        "paths": context.paths.to_json_dict(),
    }

    write_json(
        context.paths.inference_manifest_path,
        {
            "metadata": build_stage_metadata(
                config,
                "infer",
                extra={**metadata_extra, "show_inner_progress": show_inner_progress},
            ),
            "experiments_source": str(context.paths.staged_experiments_path),
            "compile_manifest_source": str(context.paths.compile_manifest_path),
        },
    )
    write_json(
        context.paths.inference_runs_path,
        {
            "metadata": build_stage_metadata(config, "infer_runs", extra=metadata_extra),
            "runs": raw_runs,
        },
    )
    save_config(config, context.paths.experiment_root / "infer_config_used.yaml")
    print(f"Saved inference manifest to: {context.paths.inference_manifest_path}")
    print(f"Saved raw inference runs to: {context.paths.inference_runs_path}")


def main() -> None:
    run_configured_stage_cli(
        run_inference_stage,
        description="Run SPLL posterior inference for staged MNIST addition experiments.",
        config_help="Path to the shared YAML config.",
    )


if __name__ == "__main__":
    main()
