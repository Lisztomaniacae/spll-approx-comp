from __future__ import annotations

import argparse
from typing import Any, Dict

from mnist_spll_common import load_config, set_seed, stage_message
from mnist_spll_pipeline_core import (
    build_pipeline_context,
    build_stage_metadata,
    get_configured_term_counts,
    sample_and_save_staged_experiments,
    stage_config_snapshot,
    write_json,
)


def run_stage_experiments(config: Dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    ctx = build_pipeline_context(config)

    stage_message(1, 2, "Sampling MNIST addition experiments from the fixed inference split")
    experiments = sample_and_save_staged_experiments(config, ctx)

    stage_message(2, 2, "Writing staged experiment bundle")
    payload = {
        "metadata": build_stage_metadata(
            config,
            "stage",
            extra={
                "num_experiments": int(ctx.inference_cfg.get("num_experiments", 200)),
                "term_counts_configured": get_configured_term_counts(config),
                "sample_without_replacement_within_experiment": bool(
                    ctx.inference_cfg.get("sample_without_replacement_within_experiment", True)
                ),
                "paths": ctx.paths.to_json_dict(),
            },
        ),
        "experiments": experiments,
    }
    write_json(ctx.paths.staged_experiments_path, payload)
    stage_config_snapshot(config, ctx.paths.experiment_root / "stage_config_used.yaml")
    print(f"Saved staged experiments to: {ctx.paths.staged_experiments_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample and save MNIST addition experiments for later SPLL inference.")
    parser.add_argument("--config", required=True, help="Path to the shared YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_stage_experiments(config)


if __name__ == "__main__":
    main()
