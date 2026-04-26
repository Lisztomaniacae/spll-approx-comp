from __future__ import annotations

import argparse
from importlib import import_module
from typing import Callable, Dict, Tuple

from mnist_spll_common import load_config


StageFn = Callable[[dict], None]
StageSpec = Tuple[str, str]


STAGES: Dict[str, StageSpec] = {
    "train": ("train_mnist", "run_training"),
    "compile": ("compile_spll", "run_compile_stage"),
    "stage": ("stage_experiments", "run_stage_experiments"),
    "infer": ("infer_experiments", "run_inference_stage"),
    "visualize": ("visualize_results", "run_visualization_stage"),
}

ORDER = ["train", "compile", "stage", "infer", "visualize"]


def load_stage_fn(stage_name: str) -> StageFn:
    """Import only the selected stage and its dependencies.

    The compile stage is often run from a different environment than train/infer
    on Apple Silicon. Lazy loading keeps that stage from importing torch,
    torchvision, matplotlib, or other dependencies owned by unrelated stages.
    """

    module_name, function_name = STAGES[stage_name]
    module = import_module(module_name)
    return getattr(module, function_name)


def run_all(config: dict) -> None:
    for stage_name in ORDER:
        print(f"\n=== Running stage: {stage_name} ===", flush=True)
        load_stage_fn(stage_name)(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one stage of the MNIST + SPLL pipeline, or run all stages in sequence. "
            "On Apple Silicon, keep using python3 for native arm stages and python under Rosetta for compile."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the shared YAML config.")
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["all", *ORDER],
        help="Which pipeline stage to run. Default: all",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.stage == "all":
        run_all(config)
    else:
        load_stage_fn(args.stage)(config)


if __name__ == "__main__":
    main()
