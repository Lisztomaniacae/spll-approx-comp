from __future__ import annotations

import sys
import unittest
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from inference_engine import (
    READ_MNIST_CACHE_POLICY_PRECOMPUTED,
    READ_MNIST_CACHE_POLICY_RUN_SCOPED,
    READ_MNIST_CACHE_POLICY_UNCACHED,
    PrecomputedReadMNistLookup,
    RunScopedReadMNistCache,
    UncachedReadMNistCounter,
    normalize_read_mnist_cache_policy,
    read_mnist_model_evaluations,
)
from pipeline1_config import get_thresholds
from pipeline1_models import get_model_variants
from pipeline2_config import get_inference_modes
from spll_artifacts import build_compile_command, extract_branch_count, extract_probability, make_spll_program


class TupleLike:
    def __init__(self, first, second):
        self.t1 = first
        self.t2 = second


class SpllContractTests(unittest.TestCase):
    def test_generated_program_text_is_stable(self) -> None:
        self.assertEqual(
            make_spll_program(3),
            "neural readMNist :: (Symbol -> Int) of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n"
            "main x0 x1 x2 = ((readMNist(x0) ++ readMNist(x1)) ++ readMNist(x2))\n",
        )
        with self.assertRaises(ValueError):
            make_spll_program(0)

    def test_compile_command_preserves_argument_order(self) -> None:
        command = build_compile_command(
            spll_path=Path("input.spll"),
            output_py_path=Path("out/program.py"),
            cutoff=0.1,
            stack_arch="x86_64",
            count_branches=True,
        )
        self.assertEqual(
            command,
            [
                "stack", "--arch", "x86_64", "run", "--", "-i", "input.spll",
                "-c", "-k", "0.1", "compile", "-o", "out/program.py", "-l", "python",
            ],
        )

    def test_generated_return_extraction_supports_tuple_and_T_shapes(self) -> None:
        self.assertEqual(extract_probability((0.25, (0, 17))), 0.25)
        self.assertEqual(extract_branch_count((0.25, (0, 17))), 17)
        generated = TupleLike(0.75, TupleLike(0, 31))
        self.assertEqual(extract_probability(generated), 0.75)
        self.assertEqual(extract_branch_count(generated), 31)
        self.assertIsNone(extract_branch_count(0.5))

    def test_pipeline1_threshold_normalization_and_order(self) -> None:
        config = {
            "inference": {
                "adaptive_top_k": {
                    "posterior_mass_target": 0.8,
                    "probe_experiments": 5,
                    "max_iterations": 7,
                    "tolerance": 0.01,
                    "min_cutoff": 0.0,
                    "max_cutoff": 0.5,
                },
                "approximation_thresholds": [
                    None,
                    0.0,
                    0.1,
                    {
                        "name": "approx_mass_0p8",
                        "top_k_cutoff": "auto",
                        "adaptive_top_k": True,
                        "posterior_mass_target": 0.8,
                    },
                ],
            }
        }
        thresholds = get_thresholds(config)
        self.assertEqual([item["threshold_label"] for item in thresholds], ["exact", "cutoff_0p0", "cutoff_0p1", "approx_mass_0p8"])
        adaptive = thresholds[-1]
        self.assertEqual(adaptive["artifact_threshold_label"], "cutoff_topk")
        self.assertEqual(adaptive["compile_cutoff"], 0.0)
        self.assertEqual(adaptive["cutoff_search"]["max_cutoff"], 0.5)

    def test_pipeline2_mode_normalization_preserves_artifact_contract(self) -> None:
        config = {
            "adaptive_top_k": {
                "posterior_mass_target": 0.8,
                "probe_cases": 5,
                "max_iterations": 8,
                "tolerance": 0.05,
                "min_cutoff": 0.0,
                "max_cutoff": 1.0,
            },
            "inference_modes": [
                {"name": "exact", "top_k_cutoff": None},
                {
                    "name": "approx_mass_0p8",
                    "top_k_cutoff": "auto",
                    "adaptive_top_k": True,
                    "posterior_mass_target": 0.8,
                },
                {"name": "approx_0p1", "top_k_cutoff": 0.1},
            ],
        }
        modes = get_inference_modes(config)
        self.assertEqual([mode["name"] for mode in modes], ["exact", "approx_mass_0p8", "approx_0p1"])
        self.assertEqual(modes[1]["artifact_name"], "cutoff_topk")
        self.assertEqual(modes[1]["top_k_cutoff"], 0.0)
        self.assertEqual(modes[2]["artifact_name"], "approx_0p1")

    def test_model_variants_are_required_and_base_model_is_merged(self) -> None:
        config = {
            "training": {
                "model": {"conv_channels": [32, 64], "dropout": 0.25},
                "epochs": 12,
                "model_variants": [
                    {"id": "small", "target_accuracy": 0.5, "model": {"conv_channels": []}},
                ],
            }
        }
        variant = get_model_variants(config)[0]
        self.assertEqual(variant["selection_mode"], "nearest")
        self.assertEqual(variant["epochs"], 12)
        self.assertEqual(variant["model"], {"conv_channels": [], "dropout": 0.25})

    def test_cache_policy_parser_accepts_only_documented_values(self) -> None:
        for policy in (
            READ_MNIST_CACHE_POLICY_UNCACHED,
            READ_MNIST_CACHE_POLICY_RUN_SCOPED,
            READ_MNIST_CACHE_POLICY_PRECOMPUTED,
        ):
            self.assertEqual(normalize_read_mnist_cache_policy(policy), policy)
        with self.assertRaises(ValueError):
            normalize_read_mnist_cache_policy("cached")

    def test_cache_implementations_have_isolated_expected_statistics(self) -> None:
        calls = []

        def base(path: str):
            calls.append(path)
            return [0.2, 0.8]

        cached = RunScopedReadMNistCache(base)
        self.assertEqual(cached("a"), [0.2, 0.8])
        self.assertEqual(cached("a"), [0.2, 0.8])
        self.assertEqual(cached.stats()["cache_hits"], 1)
        self.assertEqual(calls, ["a"])

        calls.clear()
        uncached = UncachedReadMNistCounter(base)
        uncached("a")
        uncached("a")
        self.assertEqual(uncached.stats()["cache_misses"], 2)
        self.assertEqual(calls, ["a", "a"])

        calls.clear()
        precomputed = PrecomputedReadMNistLookup(base, ["a", "a", "b"])
        self.assertEqual(calls, ["a", "b"])
        precomputed("a")
        self.assertEqual(precomputed.stats()["cache_hits"], 1)
        self.assertEqual(read_mnist_model_evaluations(precomputed.stats()), 2)
        with self.assertRaises(KeyError):
            precomputed("c")


if __name__ == "__main__":
    unittest.main()
