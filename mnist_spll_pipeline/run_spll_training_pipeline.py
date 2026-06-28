from __future__ import annotations

from typing import Dict

from pipeline_support import StageSpec, load_stage_fn as _load_stage_fn, run_pipeline_cli


STAGES: Dict[str, StageSpec] = {
    "prepare": ("prepare_spll_training", "run_prepare_stage"),
    "compile": ("compile_spll_training", "run_compile_stage"),
    "train": ("train_spll_generated", "run_train_stage"),
    "checkpoint-transfer": ("train_spll_generated", "run_checkpoint_transfer_only_stage"),
    "visualize": ("visualize_spll_training", "run_visualization_stage"),
}

# Recovery/convenience entry point; normal training already runs it last.
ORDER = ["prepare", "compile", "train", "visualize"]


def load_stage_fn(stage_name: str):
    """Import only the requested Pipeline II stage and its dependencies."""

    return _load_stage_fn(STAGES, stage_name)


def main() -> None:
    run_pipeline_cli(
        stages=STAGES,
        order=ORDER,
        description=(
            "Run one stage of the MNIST + SPLL training-through-inference pipeline. "
            "On Apple Silicon, prefer explicit stage execution because compile runs best under Rosetta/x86."
        ),
        config_help="Path to the Pipeline II YAML config.",
        heading="Running Pipeline II stage",
    )


if __name__ == "__main__":
    main()
