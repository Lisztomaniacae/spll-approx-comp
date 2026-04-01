from __future__ import annotations

import argparse
import sys

from run_spll_pipeline import ORDER, STAGES, run_all
from mnist_spll_common import load_config


LEGACY_TO_CURRENT = {
    "all": "all",
    "stage": "stage",
    "compile": "compile",
    "infer": "infer",
    "visualize": "visualize",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backward-compatible wrapper. The pipeline has been split into distinct stage files. "
            "Prefer run_spll_pipeline.py going forward."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the shared YAML config.")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=list(LEGACY_TO_CURRENT.keys()),
        help="Legacy pipeline command. Default: all",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    mapped = LEGACY_TO_CURRENT[args.command]
    if mapped == "all":
        run_all(config)
    else:
        STAGES[mapped](config)


if __name__ == "__main__":
    main()
