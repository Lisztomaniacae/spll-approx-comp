from __future__ import annotations

from typing import Any, Dict

from pipeline2_artifacts import compile_training_artifacts
from pipeline2_config import training_paths, validate_pipeline2_config
from pipeline_support import run_configured_stage_cli, save_config, stage_message


def run_compile_stage(config: Dict[str, Any]) -> None:
    validate_pipeline2_config(config)
    paths = training_paths(config)
    paths.ensure_compile_dirs()

    stage_message(1, 2, "Writing resolved Pipeline II config snapshot")
    save_config(config, paths.config_used_path)

    stage_message(2, 2, "Writing and compiling Pipeline II SPLL training artifacts")
    compile_training_artifacts(config, paths)
    print(f"Saved compiled training artifacts under: {paths.compiled_root}")


def main() -> None:
    run_configured_stage_cli(
        run_compile_stage,
        description="Compile generated SPLL artifacts for Pipeline II training.",
        config_help="Path to the Pipeline II YAML config.",
    )


if __name__ == "__main__":
    main()
