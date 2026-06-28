from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from mnist_model import CNNClassifier, set_seed
from pipeline2_config import (
    TrainingPaths,
    get_experiments,
    get_seeds,
    initial_checkpoint_path,
    schedule_manifest_path,
    schedule_preview_path,
    stable_int_seed,
    training_paths,
)
from pipeline_support import ensure_dir, utc_now_iso, write_json


def _training_transform(config: Dict[str, Any]):
    from torchvision import transforms

    normalize_cfg = config.get("data", {}).get(
        "normalize",
        {"mean": 0.1307, "std": 0.3081},
    )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (float(normalize_cfg["mean"]),),
                (float(normalize_cfg["std"]),),
            ),
        ]
    )


def load_mnist_train_dataset(config: Dict[str, Any]):
    from torchvision import datasets

    paths = training_paths(config)
    return datasets.MNIST(
        root=str(paths.data_root),
        train=True,
        download=True,
        transform=_training_transform(config),
    )


def load_mnist_train_labels(config: Dict[str, Any]) -> List[int]:
    from torchvision import datasets

    paths = training_paths(config)
    dataset = datasets.MNIST(
        root=str(paths.data_root),
        train=True,
        download=True,
        transform=None,
    )
    return [int(label) for label in dataset.targets.tolist()]


def build_balanced_split_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    split_seed = int(data_cfg.get("split_seed", 42))
    train_fraction = float(data_cfg.get("train_fraction", 0.8))
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"data.train_fraction must be in (0, 1), got {train_fraction}")

    indices_by_digit: Dict[int, List[int]] = {digit: [] for digit in range(10)}
    for index, label in enumerate(load_mnist_train_labels(config)):
        indices_by_digit[label].append(index)

    per_digit_count = min(len(indices) for indices in indices_by_digit.values())
    train_count = int(math.floor(per_digit_count * train_fraction))
    if train_count <= 0 or train_count >= per_digit_count:
        raise ValueError("Balanced split would produce an empty train or validation subset.")

    train_indices_by_digit: Dict[str, List[int]] = {}
    validation_indices_by_digit: Dict[str, List[int]] = {}
    for digit, indices in indices_by_digit.items():
        rng = np.random.default_rng(stable_int_seed("split", split_seed, digit))
        chosen = np.array(indices, dtype=np.int64)
        rng.shuffle(chosen)
        chosen = chosen[:per_digit_count]
        train_indices_by_digit[str(digit)] = [int(value) for value in chosen[:train_count]]
        validation_indices_by_digit[str(digit)] = [int(value) for value in chosen[train_count:]]

    return {
        "created_at_utc": utc_now_iso(),
        "source": "torchvision.datasets.MNIST(train=True)",
        "split_seed": split_seed,
        "train_fraction": train_fraction,
        "per_digit_count": int(per_digit_count),
        "train_count_per_digit": train_count,
        "validation_count_per_digit": int(per_digit_count - train_count),
        "train_indices_by_digit": train_indices_by_digit,
        "validation_indices_by_digit": validation_indices_by_digit,
    }


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _pool_from_manifest(
    split_manifest: Dict[str, Any],
    split: str,
    digit: int,
) -> List[int]:
    key = "train_indices_by_digit" if split == "train" else "validation_indices_by_digit"
    return [int(value) for value in split_manifest[key][str(int(digit))]]


def generate_sum_case(
    split_manifest: Dict[str, Any],
    *,
    base_seed: int,
    n_terms: int,
    step: int,
    split: str = "train",
) -> Dict[str, Any]:
    if split not in {"train", "validation"}:
        raise ValueError(f"Unsupported schedule split: {split}")

    rng = np.random.default_rng(stable_int_seed("sum-case", base_seed, n_terms, step, split))
    labels = [int(value) for value in rng.integers(0, 10, size=int(n_terms)).tolist()]
    ordered_indices: List[Optional[int]] = [None] * len(labels)

    for digit in sorted(set(labels)):
        positions = [index for index, label in enumerate(labels) if label == digit]
        pool = _pool_from_manifest(split_manifest, split, digit)
        if len(pool) < len(positions):
            raise ValueError(
                f"Digit {digit} pool for split={split} has {len(pool)} examples, "
                f"but this case needs {len(positions)} distinct examples."
            )
        chosen = rng.choice(np.array(pool, dtype=np.int64), size=len(positions), replace=False)
        for position, global_index in zip(positions, chosen.tolist()):
            ordered_indices[position] = int(global_index)

    return {
        "step": int(step),
        "n_terms": int(n_terms),
        "global_indices": [int(value) for value in ordered_indices if value is not None],
        "labels": labels,
        "true_sum": int(sum(labels)),
    }


def write_schedule_artifacts(
    config: Dict[str, Any],
    paths: TrainingPaths,
    split_manifest: Dict[str, Any],
) -> None:
    preview_size = int(config.get("schedule", {}).get("preview_size", 20))
    for seed in get_seeds(config):
        for experiment in get_experiments(config):
            n_terms = int(experiment["n_terms"])
            manifest = {
                "created_at_utc": utc_now_iso(),
                "seed": int(seed),
                "n_terms": n_terms,
                "max_steps": int(experiment["max_steps"]),
                "sampling": "with_replacement_across_steps",
                "distinct_within_case": True,
                "generator": "random_access_per_step_stable_hash",
                "split_manifest": str(paths.split_manifest_path),
            }
            write_json(schedule_manifest_path(paths, seed, n_terms), manifest)
            rows = (
                generate_sum_case(
                    split_manifest,
                    base_seed=seed,
                    n_terms=n_terms,
                    step=step,
                    split="train",
                )
                for step in range(1, preview_size + 1)
            )
            save_jsonl(schedule_preview_path(paths, seed, n_terms), rows)


def build_model_from_config(config: Dict[str, Any]) -> CNNClassifier:
    return CNNClassifier(config.get("model", {}))


def write_initial_checkpoints(config: Dict[str, Any], paths: TrainingPaths) -> None:
    for seed in get_seeds(config):
        for experiment in get_experiments(config):
            n_terms = int(experiment["n_terms"])
            set_seed(stable_int_seed("init", seed, n_terms))
            model = build_model_from_config(config)
            target = initial_checkpoint_path(paths, seed, n_terms)
            ensure_dir(target.parent)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": dict(config.get("model", {})),
                    "seed": int(seed),
                    "n_terms": n_terms,
                    "created_at_utc": utc_now_iso(),
                },
                target,
            )
