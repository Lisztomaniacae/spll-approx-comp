from __future__ import annotations

from typing import Any, Dict

from mnist_model import set_seed
from pipeline2_config import training_paths, validate_pipeline2_config
from pipeline2_data import (
    build_balanced_split_manifest,
    write_initial_checkpoints,
    write_schedule_artifacts,
)
from pipeline_support import (
    run_configured_stage_cli,
    save_config,
    stage_message,
    write_json,
)


def run_prepare_stage(config: Dict[str, Any]) -> None:
    validate_pipeline2_config(config)
    set_seed(int(config.get("seed", 42)))
    paths = training_paths(config)
    paths.ensure_prepare_dirs()

    stage_message(1, 4, "Writing resolved Pipeline II config snapshot")
    save_config(config, paths.config_used_path)

    stage_message(2, 4, "Creating balanced 80/20 MNIST-train split manifest")
    split_manifest = build_balanced_split_manifest(config)
    write_json(paths.split_manifest_path, split_manifest)
    print(f"Saved split manifest to: {paths.split_manifest_path}")

    stage_message(3, 4, "Writing compact deterministic schedule manifests and previews")
    write_schedule_artifacts(config, paths, split_manifest)
    print(f"Saved schedule manifests under: {paths.schedules_root}")

    stage_message(4, 4, "Writing shared initial checkpoints per seed and arity")
    write_initial_checkpoints(config, paths)
    print(f"Saved initial checkpoints under: {paths.initial_checkpoints_root}")


def main() -> None:
    run_configured_stage_cli(
        run_prepare_stage,
        description="Prepare data splits, schedules, and initial checkpoints for Pipeline II.",
        config_help="Path to the Pipeline II YAML config.",
    )


if __name__ == "__main__":
    main()
