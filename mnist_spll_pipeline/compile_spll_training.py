from __future__ import annotations

from typing import Any, Dict

from mnist_spll_common import load_config, save_config, set_seed, stage_message
from spll_training_core import compile_training_artifacts, training_paths


def run_compile_stage(config: Dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    paths = training_paths(config)

    stage_message(1, 2, "Writing resolved Pipeline II config snapshot")
    save_config(config, paths.config_used_path)

    stage_message(2, 2, "Writing and compiling Pipeline II SPLL training artifacts")
    compile_training_artifacts(config, paths)
    print(f"Saved compiled training artifacts under: {paths.compiled_root}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compile generated SPLL artifacts for Pipeline II training.")
    parser.add_argument("--config", required=True, help="Path to the Pipeline II YAML config.")
    args = parser.parse_args()
    run_compile_stage(load_config(args.config))


if __name__ == "__main__":
    main()
