from __future__ import annotations

from typing import Dict

from pipeline_support import StageSpec, load_stage_fn as _load_stage_fn, run_pipeline_cli


STAGES: Dict[str, StageSpec] = {
    "train": ("train_mnist", "run_training"),
    "compile": ("compile_spll", "run_compile_stage"),
    "stage": ("stage_experiments", "run_stage_experiments"),
    "infer": ("infer_experiments", "run_inference_stage"),
    "visualize": ("visualize_results", "run_visualization_stage"),
}
ORDER = ["train", "compile", "stage", "infer", "visualize"]


def load_stage_fn(stage_name: str):
    """Import only the requested stage and its dependencies."""

    return _load_stage_fn(STAGES, stage_name)


def main() -> None:
    run_pipeline_cli(
        stages=STAGES,
        order=ORDER,
        description=(
            "Run one stage of the MNIST + SPLL pipeline, or run all stages in sequence. "
            "On Apple Silicon, keep using python3 for native arm stages and python under Rosetta for compile."
        ),
        config_help="Path to the shared YAML config.",
        heading="Running stage",
    )


if __name__ == "__main__":
    main()
