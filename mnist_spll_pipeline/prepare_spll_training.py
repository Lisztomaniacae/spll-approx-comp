from __future__ import annotations

from typing import Any, Dict

from mnist_spll_common import load_config, save_config, set_seed, stage_message
from spll_training_core import (
    build_balanced_split_manifest,
    training_paths,
    write_initial_checkpoints,
    write_json,
    write_schedule_artifacts,
)


def run_prepare_stage(config: Dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    paths = training_paths(config)

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
    import argparse

    parser = argparse.ArgumentParser(description="Prepare data splits, schedules, and initial checkpoints for Pipeline II.")
    parser.add_argument("--config", required=True, help="Path to the Pipeline II YAML config.")
    args = parser.parse_args()
    run_prepare_stage(load_config(args.config))


if __name__ == "__main__":
    main()
