from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from mnist_model import compute_split_lengths
from pipeline2_checkpoint_transfer import _full_window_rolling_mean_float
from pipeline2_data import generate_sum_case
from pipeline2_runtime import (
    DirectReadMNIST,
    _as_probability_tensor,
    train_sum_batch,
    validate_probability_and_loss,
    zero_anchor_from_model,
)


class DeterminismAndRuntimeTests(unittest.TestCase):
    @staticmethod
    def split_manifest():
        return {
            "train_indices_by_digit": {
                str(digit): list(range(digit * 100, digit * 100 + 20))
                for digit in range(10)
            },
            "validation_indices_by_digit": {
                str(digit): list(range(digit * 100 + 50, digit * 100 + 70))
                for digit in range(10)
            },
        }

    def test_split_lengths_preserve_remainder_in_inference_partition(self) -> None:
        self.assertEqual(compute_split_lengths(101, 0.5, 0.4, 0.1), (50, 40, 11))
        with self.assertRaises(ValueError):
            compute_split_lengths(100, 0.5, 0.5, 0.1)

    def test_sum_cases_are_random_access_deterministic(self) -> None:
        manifest = self.split_manifest()
        case = generate_sum_case(manifest, base_seed=42, n_terms=4, step=17, split="train")
        self.assertEqual(
            case,
            {
                "step": 17,
                "n_terms": 4,
                "global_indices": [815, 512, 914, 711],
                "labels": [8, 5, 9, 7],
                "true_sum": 29,
            },
        )
        self.assertEqual(
            case,
            generate_sum_case(manifest, base_seed=42, n_terms=4, step=17, split="train"),
        )
        self.assertNotEqual(
            case,
            generate_sum_case(manifest, base_seed=42, n_terms=4, step=18, split="train"),
        )

    def test_full_window_rolling_mean_never_uses_partial_prefix(self) -> None:
        self.assertEqual(
            _full_window_rolling_mean_float([1.0, 2.0, 3.0, 4.0], 3),
            [None, None, 2.0, 3.0],
        )
        self.assertEqual(_full_window_rolling_mean_float([1.0, 2.0], 1), [1.0, 2.0])

    def test_pruned_zero_remains_connected_to_model_graph(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        anchor = zero_anchor_from_model(model)
        probability = _as_probability_tensor(0.0, allow_pruned_zero=True, zero_anchor=anchor)
        loss = probability + model.weight.sum() * 0.0
        loss.backward()
        self.assertTrue(probability.requires_grad)
        self.assertIsNotNone(model.weight.grad)

    def test_probability_validation_rejects_invalid_values(self) -> None:
        validate_probability_and_loss(torch.tensor(0.5), torch.tensor(0.7), step=1)
        with self.assertRaises(FloatingPointError):
            validate_probability_and_loss(torch.tensor(-0.1), torch.tensor(0.7), step=2)
        with self.assertRaises(FloatingPointError):
            validate_probability_and_loss(torch.tensor(0.5), torch.tensor(float("nan")), step=3)


class CheckpointAndTrainingTests(unittest.TestCase):
    def test_aggregate_checkpoint_crossings_use_across_seed_full_windows(self) -> None:
        import csv
        import tempfile

        from pipeline2_checkpoint_transfer import _build_aggregate_exact_checkpoints
        from pipeline2_config import run_dir, training_paths

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "_config_dir": str(root),
                "_config_path": str(root / "config.yaml"),
                "paths": {"output_root": "outputs"},
                "seeds": [1, 2],
                "training": {"max_steps": 4},
                "experiments": [{"n_terms": 2, "max_steps": 4}],
                "inference_modes": [{"name": "exact", "top_k_cutoff": None}],
                "checkpointing": {
                    "enabled": True,
                    "posterior_thresholds": [0.4, 0.6],
                    "rolling_window_updates": 2,
                    "rolling_window_policy": "full",
                },
            }
            paths = training_paths(config)
            masses = {1: [0.1, 0.3, 0.5, 0.7], 2: [0.2, 0.4, 0.6, 0.8]}
            for seed, values in masses.items():
                trace = run_dir(paths, seed, 2, "exact") / "train_trace.csv"
                trace.parent.mkdir(parents=True, exist_ok=True)
                with trace.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["step", "elapsed_seconds", "loss", "true_mass"])
                    writer.writeheader()
                    for step, mass in enumerate(values, start=1):
                        writer.writerow({"step": step, "elapsed_seconds": step * 0.1, "loss": 1.0 - mass, "true_mass": mass})

            payload = _build_aggregate_exact_checkpoints(
                config,
                n_terms=2,
                anchor_mode_name="exact",
                include_final_checkpoint=True,
            )

            self.assertEqual([row["step"] for row in payload["trace"]], [2, 3, 4])
            self.assertAlmostEqual(payload["trace"][0]["rolling_true_mass"], 0.25)
            self.assertEqual(payload["posterior_checkpoints"]["0.40"]["step"], 3)
            self.assertEqual(payload["posterior_checkpoints"]["0.60"]["step"], 4)
            self.assertEqual([anchor["step"] for anchor in payload["anchors"]], [0, 3, 4])

    def test_direct_read_mnist_repeats_dataset_access_and_model_forward(self) -> None:
        class CountingDataset:
            def __init__(self) -> None:
                self.get_count = 0
                self.image = torch.ones((1, 28, 28), dtype=torch.float32)

            def __getitem__(self, index: int):
                self.get_count += 1
                return self.image.clone(), 3

        class CountingModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.forward_count = 0
                self.linear = torch.nn.Linear(28 * 28, 10)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                self.forward_count += 1
                return self.linear(inputs.flatten(1))

        dataset = CountingDataset()
        model = CountingModel()
        callback = DirectReadMNIST(model, dataset, torch.device("cpu"))
        first = callback(7)
        second = callback(7)

        self.assertEqual(dataset.get_count, 2)
        self.assertEqual(model.forward_count, 2)
        self.assertEqual(callback.stats()["calls"], 2)
        self.assertEqual(callback.stats()["model_evaluations"], 2)
        self.assertEqual(callback.stats()["unique_indices"], 1)
        self.assertTrue(first.requires_grad)
        self.assertTrue(second.requires_grad)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_pruned_generated_execution_reduces_real_model_evaluations(self) -> None:
        from types import SimpleNamespace

        class TinyDataset:
            def __init__(self) -> None:
                self.image = torch.ones((1, 28, 28), dtype=torch.float32)

            def __getitem__(self, index: int):
                return self.image, 0

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(28 * 28, 10)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.linear(inputs.flatten(1))

        def make_module(call_count: int):
            module = SimpleNamespace()

            class FakeMain:
                def forward(self, sample, acc_prob, x0, x1):
                    probabilities = []
                    for idx in range(call_count):
                        image_index = x0 if idx % 2 == 0 else x1
                        probabilities.append(module.readMNist(image_index)[0])
                    probability = torch.stack(probabilities).mean()
                    return probability, call_count

            module.main = FakeMain()
            return module

        case = {"step": 1, "true_sum": 0, "global_indices": [0, 1]}
        results = {}
        for label, generated_calls in (("exact", 8), ("cutoff", 3)):
            torch.manual_seed(11)
            model = TinyModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
            callback = DirectReadMNIST(model, TinyDataset(), torch.device("cpu"))
            results[label] = train_sum_batch(
                module=make_module(generated_calls),
                model=model,
                optimizer=optimizer,
                read_mnist=callback,
                cases=[case],
                loss_epsilon=1.0e-12,
            )

        self.assertEqual(results["exact"]["read_mnist_calls_total"], 8)
        self.assertEqual(results["exact"]["read_mnist_model_evaluations_total"], 8)
        self.assertEqual(results["cutoff"]["read_mnist_calls_total"], 3)
        self.assertEqual(results["cutoff"]["read_mnist_model_evaluations_total"], 3)
        self.assertLess(
            results["cutoff"]["read_mnist_model_evaluations_total"],
            results["exact"]["read_mnist_model_evaluations_total"],
        )

    def test_fixed_budget_training_coordinator_smoke(self) -> None:
        import csv
        import json
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch

        from pipeline2_config import (
            get_experiments,
            get_inference_modes,
            initial_checkpoint_path,
            run_dir,
            training_paths,
        )
        from pipeline2_data import build_model_from_config
        from train_spll_generated import _train_one_run

        class TinyDataset:
            def __init__(self) -> None:
                generator = torch.Generator().manual_seed(123)
                self.items = [torch.rand((1, 28, 28), generator=generator) for _ in range(20)]

            def __getitem__(self, index: int):
                return self.items[int(index)], int(index) // 2

            def __len__(self) -> int:
                return len(self.items)

        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "_config_path": str(Path(tmp) / "config.yaml"),
                "_config_dir": str(Path(tmp)),
                "paths": {"output_root": str(Path(tmp) / "outputs")},
                "seeds": [7],
                "experiments": [{"n_terms": 2, "max_steps": 2}],
                "model": {
                    "input_channels": 1,
                    "conv_channels": [2, 4],
                    "kernel_size": 3,
                    "pool_kernel": 2,
                    "fc_hidden": 8,
                    "dropout": 0.0,
                    "num_classes": 10,
                },
                "optimizer": {"learning_rate": 0.001, "weight_decay": 0.0},
                "training": {
                    "device": "cpu",
                    "require_mps": False,
                    "sum_batch_size": 1,
                    "loss_epsilon": 1.0e-12,
                },
                "checkpointing": {
                    "enabled": False,
                    "posterior_thresholds": [0.2],
                    "rolling_window_updates": 2,
                    "rolling_window_policy": "full",
                },
                "inference_modes": [{"name": "exact", "top_k_cutoff": None}],
                "show_progress": False,
            }
            paths = training_paths(config)
            checkpoint_path = initial_checkpoint_path(paths, 7, 2)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": build_model_from_config(config).state_dict()}, checkpoint_path)

            split_manifest = {
                "train_indices_by_digit": {str(digit): [2 * digit, 2 * digit + 1] for digit in range(10)},
                "validation_indices_by_digit": {str(digit): [2 * digit, 2 * digit + 1] for digit in range(10)},
            }
            module = SimpleNamespace()

            class FakeMain:
                def forward(self, sample, acc_prob, x0, x1):
                    left = module.readMNist(x0)
                    right = module.readMNist(x1)
                    probability = left.new_zeros(())
                    for digit in range(10):
                        complement = int(sample) - digit
                        if 0 <= complement < 10:
                            probability = probability + left[digit] * right[complement]
                    return probability, 100

            module.main = FakeMain()
            experiment = get_experiments(config)[0]
            mode = get_inference_modes(config)[0]
            with patch("train_spll_generated.load_compiled_training_module", return_value=module):
                _train_one_run(
                    config=config,
                    split_manifest=split_manifest,
                    dataset=TinyDataset(),
                    seed=7,
                    experiment=experiment,
                    mode=mode,
                )

            output_dir = run_dir(paths, 7, 2, "exact")
            with (output_dir / "train_trace.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            summary = json.loads((output_dir / "run_summary.json").read_text())

            self.assertEqual([int(row["step"]) for row in rows], [1, 2])
            self.assertEqual(summary["completed_training_cases"], 2)
            self.assertEqual(summary["optimizer_updates"], 2)
            self.assertEqual(summary["mode_name"], "exact")
            self.assertEqual(summary["read_mnist_policy"], "direct_uncached_model_forward_per_generated_call")
            self.assertEqual([int(row["read_mnist_calls_total"]) for row in rows], [2, 2])
            self.assertEqual([int(row["read_mnist_model_evaluations_total"]) for row in rows], [2, 2])
            self.assertTrue((output_dir / "checkpoints" / "final.pt").exists())
            self.assertFalse((output_dir / "failure_report.json").exists())

    def test_training_checkpoint_schema_is_preserved(self) -> None:
        import tempfile

        from pipeline2_runtime import save_training_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            mode = {"name": "exact", "artifact_name": "exact", "top_k_cutoff": None}
            save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                step=7,
                elapsed_seconds=1.25,
                digit_accuracy_value=0.6,
                config={"model": {"input": 2}, "_config_path": "/tmp/config.yaml"},
                seed=42,
                n_terms=2,
                mode=mode,
                validation_metric_name="true_sum_posterior_rolling_mean",
                sum_posterior_accuracy=0.6,
            )
            payload = torch.load(path, map_location="cpu")
            expected = {
                "state_dict", "optimizer_state_dict", "step", "elapsed_seconds",
                "digit_accuracy", "validation_metric_name", "sum_posterior_accuracy",
                "model_config", "seed", "n_terms", "inference_mode", "artifact_name",
                "top_k_cutoff", "read_mnist_policy", "config_path", "created_at_utc",
            }
            self.assertEqual(set(payload), expected)
            self.assertEqual(payload["step"], 7)
            self.assertEqual(payload["digit_accuracy"], 0.6)


if __name__ == "__main__":
    unittest.main()
