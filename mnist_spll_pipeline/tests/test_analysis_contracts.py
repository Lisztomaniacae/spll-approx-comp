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
from pipeline2_plotting import (
    _checkpoint_bar_axis_scaling,
    _fully_reached_milestone_rows,
    _mean_trajectory_milestone_intervals_from_rows,
    _reached_milestone_rows,
    _project_checkpoint_values_onto_displayed_series,
    _require_checkpoint_transfer_v2_rows,
)


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
                "read_mnist_cache_policy": "uncached",
                "posterior_read_mnist_stats": {
                    "policy": "uncached",
                    "calls": 100,
                    "cache_hits": 0,
                    "cache_misses": 100,
                    "unique_images": 2,
                },
                "true_candidate_read_mnist_stats": {
                    "policy": "uncached",
                    "calls": 2,
                    "cache_hits": 0,
                    "cache_misses": 2,
                    "unique_images": 2,
                },
            },
            {
                **common,
                "threshold_label": "cutoff_0p1",
                "cutoff": 0.1,
                "runtime_sec": 1.0,
                "posterior_raw": approx_posterior,
                "true_candidate_probability_raw": 0.0,
                "read_mnist_cache_policy": "uncached",
                "posterior_read_mnist_stats": {
                    "policy": "uncached",
                    "calls": 40,
                    "cache_hits": 0,
                    "cache_misses": 40,
                    "unique_images": 2,
                },
                "true_candidate_read_mnist_stats": {
                    "policy": "uncached",
                    "calls": 1,
                    "cache_hits": 0,
                    "cache_misses": 1,
                    "unique_images": 1,
                },
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
        self.assertEqual(rows["exact"]["mean_read_mnist_lookup_calls"], 100.0)
        self.assertEqual(rows["cutoff_0p1"]["mean_read_mnist_lookup_calls"], 40.0)
        self.assertAlmostEqual(rows["cutoff_0p1"]["read_mnist_lookup_ratio_vs_exact"], 0.4)
        self.assertAlmostEqual(rows["cutoff_0p1"]["read_mnist_lookup_reduction_vs_exact"], 0.6)
        self.assertAlmostEqual(rows["cutoff_0p1"]["read_mnist_lookup_speedup_vs_exact"], 2.5)

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

    def test_checkpoint_transfer_plot_rejects_early_stopped_segments(self) -> None:
        fixed_row = {
            "segment_index": "3",
            "target_step": "2200",
            "max_segment_cases_exact": "150",
            "actual_end_step": "2200",
            "reached_target_checkpoint": "True",
        }
        _require_checkpoint_transfer_v2_rows(Path("fixed.csv"), [fixed_row])

        stale_row = dict(fixed_row, actual_end_step="2150")
        with self.assertRaisesRegex(RuntimeError, "early-stopped checkpoint-transfer trace"):
            _require_checkpoint_transfer_v2_rows(Path("stale.csv"), [stale_row])

    def test_milestone_bars_require_every_configured_seed_to_reach_target(self) -> None:
        config = {"seeds": [11, 22]}
        rows = [
            {
                "seed": 11,
                "mode_name": "approx_0p1",
                "milestone": 0.8,
                "reached": True,
                "step": 2400,
                "elapsed_seconds": 140.0,
            },
            {
                "seed": 22,
                "mode_name": "approx_0p1",
                "milestone": 0.8,
                "reached": False,
                "step": None,
                "elapsed_seconds": None,
            },
        ]

        self.assertEqual(
            _fully_reached_milestone_rows(
                config,
                rows,
                mode_name="approx_0p1",
                milestone=0.8,
                required_fields=("step", "elapsed_seconds"),
            ),
            [],
        )
        successful_subset = _reached_milestone_rows(
            config,
            rows,
            mode_name="approx_0p1",
            milestone=0.8,
            required_fields=("step", "elapsed_seconds"),
        )
        self.assertEqual([row["seed"] for row in successful_subset], [11])

        rows[1].update(reached=True, step=2500, elapsed_seconds=145.0)
        complete = _fully_reached_milestone_rows(
            config,
            rows,
            mode_name="approx_0p1",
            milestone=0.8,
            required_fields=("step", "elapsed_seconds"),
        )
        self.assertEqual([row["seed"] for row in complete], [11, 22])

    def test_mean_trajectory_milestones_average_before_crossing(self) -> None:
        per_seed_rows = {
            11: [
                {"step": 1, "true_mass": 0.10, "elapsed_seconds": 1.0},
                {"step": 2, "true_mass": 0.30, "elapsed_seconds": 2.0},
                {"step": 3, "true_mass": 0.50, "elapsed_seconds": 3.0},
                {"step": 4, "true_mass": 0.70, "elapsed_seconds": 4.0},
                {"step": 5, "true_mass": 0.90, "elapsed_seconds": 5.0},
            ],
            22: [
                {"step": 1, "true_mass": 0.10, "elapsed_seconds": 1.5},
                {"step": 2, "true_mass": 0.10, "elapsed_seconds": 3.0},
                {"step": 3, "true_mass": 0.30, "elapsed_seconds": 4.5},
                {"step": 4, "true_mass": 0.50, "elapsed_seconds": 6.0},
                {"step": 5, "true_mass": 0.70, "elapsed_seconds": 7.5},
            ],
        }
        intervals = _mean_trajectory_milestone_intervals_from_rows(
            per_seed_rows,
            milestones=[0.2, 0.4, 0.6],
            rolling_window=1,
            secondary_key="elapsed_seconds",
        )

        # Mean posterior is [0.10, 0.20, 0.40, 0.60, 0.80], so crossings
        # occur at steps 2, 3, and 4.  The wall-clock trace is averaged at
        # those same common steps before interval differencing.
        self.assertEqual(intervals[0.2]["crossing_step"], 2)
        self.assertEqual(intervals[0.4]["crossing_step"], 3)
        self.assertEqual(intervals[0.6]["crossing_step"], 4)
        self.assertAlmostEqual(intervals[0.2]["step_delta"], 2.0)
        self.assertAlmostEqual(intervals[0.4]["step_delta"], 1.0)
        self.assertAlmostEqual(intervals[0.6]["step_delta"], 1.0)
        self.assertAlmostEqual(intervals[0.2]["secondary_delta"], 2.5)
        self.assertAlmostEqual(intervals[0.4]["secondary_delta"], 1.25)
        self.assertAlmostEqual(intervals[0.6]["secondary_delta"], 1.25)
        self.assertEqual(intervals[0.6]["seed_count"], 2)

    def test_mean_trajectory_milestones_mark_unreached_endpoint(self) -> None:
        per_seed_rows = {
            11: [
                {"step": 1, "true_mass": 0.10, "elapsed_seconds": 1.0},
                {"step": 2, "true_mass": 0.30, "elapsed_seconds": 2.0},
            ],
            22: [
                {"step": 1, "true_mass": 0.10, "elapsed_seconds": 1.0},
                {"step": 2, "true_mass": 0.20, "elapsed_seconds": 2.0},
            ],
        }
        intervals = _mean_trajectory_milestone_intervals_from_rows(
            per_seed_rows,
            milestones=[0.2, 0.4],
            rolling_window=1,
            secondary_key="elapsed_seconds",
        )
        self.assertTrue(intervals[0.2]["reached"])
        self.assertFalse(intervals[0.4]["reached"])
        self.assertIsNone(intervals[0.4]["step_delta"])
        self.assertIsNone(intervals[0.4]["secondary_delta"])

    def test_checkpoint_markers_follow_the_display_smoothing_window(self) -> None:
        marker_xs, marker_ys = _project_checkpoint_values_onto_displayed_series(
            checkpoint_xs=[500.0, 900.0],
            checkpoint_ys=[0.20, 0.60],
            displayed_xs=[500.0, 700.0, 900.0],
            displayed_ys=[0.17, 0.36, 0.55],
        )

        self.assertEqual(marker_xs, [500.0, 900.0])
        self.assertEqual(marker_ys, [0.17, 0.55])

    def test_dual_axis_checkpoint_plot_budgets_headroom_for_inner_time_bars(self) -> None:
        time_to_steps, left_top = _checkpoint_bar_axis_scaling(
            max_step_value=2100.0,
            max_time_value=150.0,
            exact_step_means=[540.0, 690.0, 930.0, 1590.0],
            exact_time_means=[30.0, 38.0, 52.0, 88.0],
        )

        self.assertGreater(time_to_steps, 0.0)
        self.assertGreater(150.0 * time_to_steps, 2100.0)
        self.assertGreater(left_top, 150.0 * time_to_steps)


if __name__ == "__main__":
    unittest.main()
