from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline1_analysis import (
    add_exact_baseline_columns,
    entropy_from_distribution,
    normalize_distribution,
    prepare_detailed_rows,
    summarize_groups,
    top_predictions,
)
from pipeline2_analysis import _uncertainty_half_width


class AnalysisContractTests(unittest.TestCase):
    def test_distribution_helpers(self) -> None:
        self.assertEqual(normalize_distribution([2.0, 2.0]), [0.5, 0.5])
        self.assertEqual(normalize_distribution([0.0, 0.0]), [0.0, 0.0])
        self.assertAlmostEqual(entropy_from_distribution([0.5, 0.5]), math.log(2.0))
        self.assertEqual(top_predictions([0.1, 0.7, 0.2], 2), [
            {"sum": 1, "probability": 0.7},
            {"sum": 2, "probability": 0.2},
        ])

    def test_detailed_rows_and_exact_baseline_deltas(self) -> None:
        common = {
            "model_id": "acc70",
            "visualization_group": "main",
            "target_accuracy": 0.7,
            "selected_epoch": 2,
            "selected_test_accuracy": 0.71,
            "experiment_id": 1,
            "n_terms": 2,
            "true_sum": 3,
            "candidate_sums": list(range(19)),
            "branch_counts_raw": [1] * 19,
            "true_candidate_sum": 3,
            "true_candidate_branch_count": 1,
            "true_candidate_runtime_sec": 0.01,
            "labels": [1, 2],
            "global_indices": [10, 20],
            "image_paths": ["a.png", "b.png"],
            "cutoff_mode": "global",
        }
        exact_posterior = [0.0] * 19
        exact_posterior[3] = 1.0
        approx_posterior = [0.0] * 19
        approx_posterior[4] = 1.0
        runs = [
            {
                **common,
                "threshold_label": "exact",
                "cutoff": None,
                "runtime_sec": 2.0,
                "posterior_raw": exact_posterior,
                "true_candidate_probability_raw": 1.0,
            },
            {
                **common,
                "threshold_label": "cutoff_0p1",
                "cutoff": 0.1,
                "runtime_sec": 1.0,
                "posterior_raw": approx_posterior,
                "true_candidate_probability_raw": 0.0,
            },
        ]
        detailed = prepare_detailed_rows(runs, top_n=3)
        summary = summarize_groups(
            detailed,
            group_keys=["cutoff_mode", "model_id", "n_terms", "threshold_label", "cutoff"],
            threshold_order=["exact", "cutoff_0p1"],
        )
        add_exact_baseline_columns(
            summary,
            ["cutoff_mode", "model_id", "n_terms", "threshold_label", "cutoff"],
        )
        rows = {row["threshold_label"]: row for row in summary}
        self.assertEqual(rows["exact"]["accuracy"], 1.0)
        self.assertEqual(rows["cutoff_0p1"]["accuracy"], 0.0)
        self.assertEqual(rows["cutoff_0p1"]["speedup_vs_exact"], 2.0)
        self.assertEqual(rows["cutoff_0p1"]["accuracy_delta_vs_exact"], -1.0)

    def test_pipeline2_uncertainty_modes(self) -> None:
        values = [1.0, 2.0, 3.0]
        base = {"visualization": {"min_uncertainty_samples": 2}}
        self.assertAlmostEqual(
            _uncertainty_half_width(values, {**base, "visualization": {**base["visualization"], "uncertainty_interval": "std"}}),
            1.0,
        )
        self.assertAlmostEqual(
            _uncertainty_half_width(values, {**base, "visualization": {**base["visualization"], "uncertainty_interval": "sem"}}),
            1.0 / math.sqrt(3.0),
        )
        self.assertIsNone(
            _uncertainty_half_width(values, {"visualization": {"uncertainty_interval": "none"}})
        )


if __name__ == "__main__":
    unittest.main()
