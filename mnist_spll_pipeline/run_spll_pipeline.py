from __future__ import annotations

import argparse
from typing import Callable, Dict

from compile_spll import run_compile_stage
from infer_experiments import run_inference_stage
from mnist_spll_common import load_config
from stage_experiments import run_stage_experiments
from train_mnist import run_training
from visualize_results import run_visualization_stage


StageFn = Callable[[dict], None]


STAGES: Dict[str, StageFn] = {
    "train": run_training,
    "compile": run_compile_stage,
    "stage": run_stage_experiments,
    "infer": run_inference_stage,
    "visualize": run_visualization_stage,
}

ORDER = ["train", "compile", "stage", "infer", "visualize"]



def run_all(config: dict) -> None:
    for stage_name in ORDER:
        print(f"\n=== Running stage: {stage_name} ===", flush=True)
        STAGES[stage_name](config)



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
        STAGES[args.stage](config)


if __name__ == "__main__":
    main()
