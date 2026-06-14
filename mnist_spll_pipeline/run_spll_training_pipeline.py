from __future__ import annotations

import argparse
from importlib import import_module
from typing import Callable, Dict, Tuple

from mnist_spll_common import load_config


StageFn = Callable[[dict], None]
StageSpec = Tuple[str, str]


STAGES: Dict[str, StageSpec] = {
    "prepare": ("prepare_spll_training", "run_prepare_stage"),
    "compile": ("compile_spll_training", "run_compile_stage"),
    "train": ("train_spll_generated", "run_train_stage"),
    "checkpoint-transfer": ("train_spll_generated", "run_checkpoint_transfer_only_stage"),
    "visualize": ("visualize_spll_training", "run_visualization_stage"),
}

# The explicit checkpoint-transfer stage is a recovery/convenience entry point.
# It is not part of ORDER because the normal train stage already runs it as its
# final substage.
ORDER = ["prepare", "compile", "train", "visualize"]


def load_stage_fn(stage_name: str) -> StageFn:
    """Import only the selected stage and its dependencies.

    Pipeline II keeps the Rosetta/x86 SPLL compile stage separate from the
    native arm64 torch-heavy training stages. Lazy loading avoids importing more
    than the selected stage needs.
    """

    module_name, function_name = STAGES[stage_name]
    module = import_module(module_name)
    return getattr(module, function_name)


def run_all(config: dict) -> None:
    for stage_name in ORDER:
        print(f"\n=== Running Pipeline II stage: {stage_name} ===", flush=True)
        load_stage_fn(stage_name)(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one stage of the MNIST + SPLL training-through-inference pipeline. "
            "On Apple Silicon, prefer explicit stage-by-stage execution because compile runs best under Rosetta/x86."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the Pipeline II YAML config.")
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["all", *STAGES.keys()],
        help="Which Pipeline II stage to run. Default: all",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.stage == "all":
        run_all(config)
    else:
        load_stage_fn(args.stage)(config)


if __name__ == "__main__":
    main()
