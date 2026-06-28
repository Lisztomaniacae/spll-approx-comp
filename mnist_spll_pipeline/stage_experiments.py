from __future__ import annotations

from typing import Any, Dict

from mnist_model import set_seed
from pipeline1_config import build_pipeline_context, get_configured_term_counts, get_term_count_bounds
from pipeline1_data import stage_experiment_bundle
from pipeline_support import (
    build_stage_metadata,
    run_configured_stage_cli,
    save_config,
    stage_message,
    write_json,
)


def run_stage_experiments(config: Dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    context = build_pipeline_context(config)
    context.paths.ensure_stage_dirs()
    terms_min, terms_max = get_term_count_bounds(config)

    stage_message(1, 2, "Sampling MNIST addition experiments from the fixed inference split")
    experiments = stage_experiment_bundle(
        config,
        inference_cfg=context.inference_cfg,
        inputs_root=context.paths.inputs_root,
        show_progress=context.show_progress,
        terms_min=terms_min,
        terms_max=terms_max,
    )

    stage_message(2, 2, "Writing staged experiment bundle")
    write_json(
        context.paths.staged_experiments_path,
        {
            "metadata": build_stage_metadata(
                config,
                "stage",
                extra={
                    "num_experiments": int(context.inference_cfg.get("num_experiments", 200)),
                    "term_counts_configured": get_configured_term_counts(config),
                    "sample_without_replacement_within_experiment": bool(
                        context.inference_cfg.get(
                            "sample_without_replacement_within_experiment",
                            True,
                        )
                    ),
                    "paths": context.paths.to_json_dict(),
                },
            ),
            "experiments": experiments,
        },
    )
    save_config(config, context.paths.experiment_root / "stage_config_used.yaml")
    print(f"Saved staged experiments to: {context.paths.staged_experiments_path}")


def main() -> None:
    run_configured_stage_cli(
        run_stage_experiments,
        description="Sample and save MNIST addition experiments for later SPLL inference.",
        config_help="Path to the shared YAML config.",
    )


if __name__ == "__main__":
    main()
