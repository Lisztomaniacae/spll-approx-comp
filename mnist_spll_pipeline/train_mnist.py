from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, random_split

from mnist_spll_common import (
    build_model,
    checkpoint_payload,
    compute_split_lengths,
    ensure_dir,
    get_model_selection_manifest_path,
    get_model_variants,
    get_models_root,
    get_training_root,
    get_variant_metrics_output_path,
    get_variant_model_output_path,
    load_config,
    load_full_mnist_transformed,
    resolve_device,
    resolve_path,
    save_config,
    set_seed,
    stable_variant_offset,
)


NUM_MNIST_CLASSES = 10


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += int(labels.size(0))
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }



def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: Adam, device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += int(labels.size(0))

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }



def choose_epoch_nearest_target(history: List[Dict[str, float]], target_accuracy: float) -> Tuple[int, Dict[str, float]]:
    if not history:
        raise RuntimeError("Cannot select an epoch from an empty training history.")
    indexed = list(enumerate(history, start=1))
    best_index, best_row = min(
        indexed,
        key=lambda item: (
            abs(float(item[1]["test_accuracy"]) - float(target_accuracy)),
            item[0],
        ),
    )
    return best_index, best_row



def within_target_tolerance(accuracy: float, target_accuracy: float, tolerance: float | None) -> bool:
    if tolerance is None:
        return False
    return abs(float(accuracy) - float(target_accuracy)) <= float(tolerance)



def resolve_requested_examples(
        total_examples: int,
        *,
        variant: Dict[str, Any],
        max_examples_key: str,
        ratio_key: str,
) -> int:
    max_examples = variant.get(max_examples_key)
    subset_ratio = variant.get(ratio_key)

    if max_examples is not None and subset_ratio is not None:
        raise ValueError(
            f"Variant '{variant['id']}' sets both {max_examples_key} and {ratio_key}; pick only one."
        )

    if subset_ratio is not None:
        requested = int(round(total_examples * float(subset_ratio)))
    elif max_examples is not None:
        requested = int(max_examples)
    else:
        requested = total_examples

    return max(1, min(total_examples, requested))



def normalize_label_distribution(
        raw_distribution: Mapping[Any, Any],
        *,
        variant_id: str,
        field_name: str,
) -> Dict[int, float]:
    if not isinstance(raw_distribution, Mapping):
        raise ValueError(f"Variant '{variant_id}' {field_name} must be a mapping.")

    normalized: Dict[int, float] = {label: 0.0 for label in range(NUM_MNIST_CLASSES)}
    for raw_label, raw_weight in raw_distribution.items():
        label = int(raw_label)
        if not (0 <= label < NUM_MNIST_CLASSES):
            raise ValueError(f"Variant '{variant_id}' {field_name} contains invalid label {raw_label!r}.")
        weight = float(raw_weight)
        if weight < 0.0:
            raise ValueError(f"Variant '{variant_id}' {field_name} contains a negative weight for label {label}.")
        normalized[label] = weight

    total_weight = sum(normalized.values())
    if total_weight <= 0.0:
        raise ValueError(f"Variant '{variant_id}' {field_name} must assign positive mass to at least one label.")
    return {label: weight / total_weight for label, weight in normalized.items()}



def allocate_label_counts(total_examples: int, distribution: Mapping[int, float]) -> Dict[int, int]:
    raw_targets = {label: float(total_examples) * float(distribution.get(label, 0.0)) for label in range(NUM_MNIST_CLASSES)}
    counts = {label: int(np.floor(raw_targets[label])) for label in range(NUM_MNIST_CLASSES)}
    remainder = int(total_examples - sum(counts.values()))
    if remainder > 0:
        fractional_order = sorted(
            range(NUM_MNIST_CLASSES),
            key=lambda label: (raw_targets[label] - counts[label], -label),
            reverse=True,
        )
        for label in fractional_order[:remainder]:
            counts[label] += 1
    return counts



def build_label_index_pools(base_subset) -> Dict[int, List[int]]:
    pools: Dict[int, List[int]] = {label: [] for label in range(NUM_MNIST_CLASSES)}
    for local_idx in range(len(base_subset)):
        _, label = base_subset[local_idx]
        pools[int(label)].append(local_idx)
    return pools



def summarize_selected_counts(indices: Sequence[int], label_pools: Dict[int, List[int]]) -> Dict[int, int]:
    reverse_lookup: Dict[int, int] = {}
    for label, pool in label_pools.items():
        for local_idx in pool:
            reverse_lookup[int(local_idx)] = int(label)

    counts = {label: 0 for label in range(NUM_MNIST_CLASSES)}
    for local_idx in indices:
        label = reverse_lookup.get(int(local_idx))
        if label is None:
            raise RuntimeError(f"Internal error: local index {local_idx} not found in label pools.")
        counts[label] += 1
    return counts



def select_variant_subset(
        base_subset,
        *,
        variant: Dict[str, Any],
        subset_name: str,
        max_examples_key: str,
        ratio_key: str,
        distribution_key: str,
        sampling_with_replacement_key: str,
        base_seed: int,
        label_index_pools: Dict[int, List[int]],
):
    total_examples = len(base_subset)
    requested = resolve_requested_examples(
        total_examples,
        variant=variant,
        max_examples_key=max_examples_key,
        ratio_key=ratio_key,
    )
    rng_offset = stable_variant_offset(f"{variant['id']}::{subset_name}")
    rng = np.random.default_rng(base_seed + rng_offset)

    raw_distribution = variant.get(distribution_key)
    sampling_with_replacement = bool(variant.get(sampling_with_replacement_key, False))

    if raw_distribution is None:
        if requested >= total_examples and not sampling_with_replacement:
            selected_indices = list(range(total_examples))
            selected_subset = base_subset
        else:
            selected_indices = rng.choice(total_examples, size=requested, replace=sampling_with_replacement).tolist()
            selected_subset = Subset(base_subset, selected_indices)

        label_counts = summarize_selected_counts(selected_indices, label_index_pools)
        return selected_subset, requested, {
            "mode": "iid_subsample" if requested < total_examples else "full_split",
            "with_replacement": sampling_with_replacement,
            "requested_label_distribution": None,
            "selected_label_counts": label_counts,
        }

    distribution = normalize_label_distribution(
        raw_distribution,
        variant_id=str(variant["id"]),
        field_name=distribution_key,
    )
    target_counts = allocate_label_counts(requested, distribution)
    selected_indices: List[int] = []

    for label in range(NUM_MNIST_CLASSES):
        needed = int(target_counts[label])
        if needed <= 0:
            continue
        candidates = label_index_pools[label]
        if not candidates:
            raise ValueError(
                f"Variant '{variant['id']}' requested label {label} in {distribution_key}, "
                f"but there are no examples for that label in the {subset_name} split."
            )
        if not sampling_with_replacement and needed > len(candidates):
            raise ValueError(
                f"Variant '{variant['id']}' requests {needed} examples for label {label} in {subset_name}, "
                f"but only {len(candidates)} are available without replacement. "
                f"Reduce {max_examples_key}, relax the skew, or enable {sampling_with_replacement_key}."
            )
        chosen = rng.choice(candidates, size=needed, replace=sampling_with_replacement).tolist()
        selected_indices.extend(int(idx) for idx in chosen)

    if len(selected_indices) != requested:
        raise RuntimeError(
            f"Internal error: selected {len(selected_indices)} {subset_name} examples, expected {requested}."
        )

    rng.shuffle(selected_indices)
    selected_subset = Subset(base_subset, selected_indices)
    selected_label_counts = {label: int(target_counts[label]) for label in range(NUM_MNIST_CLASSES)}
    return selected_subset, requested, {
        "mode": "label_skew",
        "with_replacement": sampling_with_replacement,
        "requested_label_distribution": distribution,
        "selected_label_counts": selected_label_counts,
    }



def write_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for metrics export to {path}")
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def train_variant(
        *,
        config: Dict[str, Any],
        variant: Dict[str, Any],
        train_subset,
        validation_subset,
        train_label_index_pools: Dict[int, List[int]],
        validation_label_index_pools: Dict[int, List[int]],
        device: torch.device,
        used_config_path: Path,
) -> Dict[str, Any]:
    training_cfg = config["training"]
    batch_size = int(variant.get("batch_size", training_cfg.get("batch_size", 128)))
    eval_batch_size = int(variant.get("eval_batch_size", training_cfg.get("eval_batch_size", 256)))
    num_workers = int(variant.get("num_workers", training_cfg.get("num_workers", 0)))
    learning_rate = float(variant.get("learning_rate", training_cfg.get("learning_rate", 1e-3)))
    weight_decay = float(variant.get("weight_decay", training_cfg.get("weight_decay", 0.0)))
    epochs = int(variant.get("epochs", training_cfg.get("epochs", 8)))
    target_accuracy = float(variant["target_accuracy"])
    target_tolerance_raw = variant.get("target_tolerance", training_cfg.get("target_tolerance"))
    target_tolerance = None if target_tolerance_raw is None else float(target_tolerance_raw)

    selected_train_subset, selected_train_examples, train_selection_meta = select_variant_subset(
        train_subset,
        variant=variant,
        subset_name="train",
        max_examples_key="max_train_examples",
        ratio_key="train_subset_ratio",
        distribution_key="train_label_distribution",
        sampling_with_replacement_key="train_sampling_with_replacement",
        base_seed=int(config.get("seed", 42)),
        label_index_pools=train_label_index_pools,
    )
    selected_validation_subset, selected_validation_examples, validation_selection_meta = select_variant_subset(
        validation_subset,
        variant=variant,
        subset_name="validation",
        max_examples_key="max_validation_examples",
        ratio_key="validation_subset_ratio",
        distribution_key="validation_label_distribution",
        sampling_with_replacement_key="validation_sampling_with_replacement",
        base_seed=int(config.get("seed", 42)) + 1_000_000,
        label_index_pools=validation_label_index_pools,
    )

    train_loader = DataLoader(selected_train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(selected_validation_subset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(config, model_cfg=variant["model"]).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    export_path = get_variant_model_output_path(config, variant["id"])
    metrics_path = get_variant_metrics_output_path(config, variant["id"])
    ensure_dir(export_path.parent)
    ensure_dir(metrics_path.parent)
    ensure_dir(used_config_path.parent)

    history: List[Dict[str, Any]] = []
    epoch_states: Dict[int, Dict[str, torch.Tensor]] = {}
    best_overall_accuracy = float("-inf")
    best_overall_epoch = -1
    stopped_early = False
    stop_reason = "max_epochs"

    tolerance_display = f"{target_tolerance:.1%}" if target_tolerance is not None else "disabled"
    print(
        f"\n--- Training model variant '{variant['id']}' "
        f"(target={target_accuracy:.1%}, tolerance={tolerance_display}, "
        f"train_examples={selected_train_examples}, validation_examples={selected_validation_examples}) ---"
    )
    print(
        f"train_mode={train_selection_meta['mode']} val_mode={validation_selection_meta['mode']}"
    )
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        validation_metrics = evaluate(model, validation_loader, device)
        row = {
            "model_id": variant["id"],
            "target_accuracy": target_accuracy,
            "epoch": epoch,
            "train_examples": selected_train_examples,
            "validation_examples": selected_validation_examples,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": validation_metrics["loss"],
            "test_accuracy": validation_metrics["accuracy"],
            "accuracy_gap_to_target": abs(float(validation_metrics["accuracy"]) - target_accuracy),
            "within_target_tolerance": int(within_target_tolerance(validation_metrics["accuracy"], target_accuracy, target_tolerance)),
            "target_tolerance": "" if target_tolerance is None else target_tolerance,
        }
        history.append(row)
        epoch_states[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if float(validation_metrics["accuracy"]) > best_overall_accuracy:
            best_overall_accuracy = float(validation_metrics["accuracy"])
            best_overall_epoch = epoch

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4%} | "
            f"val_loss={row['test_loss']:.4f} val_acc={row['test_accuracy']:.4%} | "
            f"target_gap={row['accuracy_gap_to_target']:.4%}"
        )

        if within_target_tolerance(validation_metrics["accuracy"], target_accuracy, target_tolerance):
            stopped_early = True
            stop_reason = "target_tolerance_reached"
            print(
                f"Early stopping '{variant['id']}' at epoch {epoch} because val_acc={row['test_accuracy']:.4%} "
                f"is within ±{target_tolerance:.1%} of target {target_accuracy:.1%}."
            )
            break

    epochs_trained = len(history)
    selected_epoch, selected_row = choose_epoch_nearest_target(history, target_accuracy)
    selected_state = epoch_states.get(selected_epoch)
    if selected_state is None:
        raise RuntimeError(f"Selected epoch {selected_epoch} is missing a saved model state.")

    model.load_state_dict(selected_state)
    checkpoint = checkpoint_payload(
        model=model,
        config={**config, "training": {**config["training"], "model": variant["model"]}},
        best_epoch=selected_epoch,
        best_test_accuracy=float(selected_row["test_accuracy"]),
        extra={
            "model_variant_id": variant["id"],
            "target_accuracy": target_accuracy,
            "selected_epoch": selected_epoch,
            "selected_test_accuracy": float(selected_row["test_accuracy"]),
            "selected_accuracy_gap": float(selected_row["accuracy_gap_to_target"]),
            "best_overall_epoch": best_overall_epoch,
            "best_overall_test_accuracy": best_overall_accuracy,
            "train_examples": selected_train_examples,
            "validation_examples": selected_validation_examples,
            "train_selection_mode": train_selection_meta["mode"],
            "validation_selection_mode": validation_selection_meta["mode"],
            "train_label_distribution": train_selection_meta["requested_label_distribution"],
            "validation_label_distribution": validation_selection_meta["requested_label_distribution"],
            "train_label_counts": train_selection_meta["selected_label_counts"],
            "validation_label_counts": validation_selection_meta["selected_label_counts"],
            "epochs_trained": epochs_trained,
            "target_tolerance": target_tolerance,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
        },
    )
    torch.save(checkpoint, export_path)
    write_metrics_csv(metrics_path, history)
    save_config(config, used_config_path)

    print(
        f"Selected epoch {selected_epoch} for '{variant['id']}' with val_acc={selected_row['test_accuracy']:.4%} "
        f"(target={target_accuracy:.4%}, gap={selected_row['accuracy_gap_to_target']:.4%})."
    )
    print(f"Saved model checkpoint to: {export_path}")
    print(f"Saved per-epoch metrics to: {metrics_path}")

    return {
        "model_id": variant["id"],
        "target_accuracy": target_accuracy,
        "selected_epoch": selected_epoch,
        "selected_test_accuracy": float(selected_row["test_accuracy"]),
        "selected_accuracy_gap": float(selected_row["accuracy_gap_to_target"]),
        "best_overall_epoch": best_overall_epoch,
        "best_overall_test_accuracy": best_overall_accuracy,
        "epochs_trained": epochs_trained,
        "target_tolerance": target_tolerance,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "train_examples": selected_train_examples,
        "validation_examples": selected_validation_examples,
        "train_selection_mode": train_selection_meta["mode"],
        "validation_selection_mode": validation_selection_meta["mode"],
        "train_label_distribution": train_selection_meta["requested_label_distribution"],
        "validation_label_distribution": validation_selection_meta["requested_label_distribution"],
        "train_label_counts": train_selection_meta["selected_label_counts"],
        "validation_label_counts": validation_selection_meta["selected_label_counts"],
        "model_output": str(export_path),
        "metrics_csv": str(metrics_path),
        "model_config": variant["model"],
    }



def run_training(config: Dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))

    training_cfg = config["training"]
    paths_cfg = config["paths"]

    device = resolve_device(training_cfg.get("device", "auto"), bool(training_cfg.get("require_mps", False)))
    print(f"Using device: {device}")

    full_dataset = load_full_mnist_transformed(config, train=True)
    total_size = len(full_dataset)
    train_len, test_len, inference_len = compute_split_lengths(
        total_size,
        float(training_cfg["train_ratio"]),
        float(training_cfg["test_ratio"]),
        float(training_cfg["inference_ratio"]),
    )
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    train_subset, validation_subset, inference_subset = random_split(
        full_dataset,
        [train_len, test_len, inference_len],
        generator=generator,
    )

    train_label_index_pools = build_label_index_pools(train_subset)
    validation_label_index_pools = build_label_index_pools(validation_subset)

    split_manifest_path = resolve_path(config, paths_cfg["split_manifest"])
    used_config_path = resolve_path(config, paths_cfg.get("used_config_copy", "./outputs/config_used.yaml"))
    manifest_path = get_model_selection_manifest_path(config)
    ensure_dir(split_manifest_path.parent)
    ensure_dir(used_config_path.parent)
    ensure_dir(manifest_path.parent)
    get_models_root(config)
    get_training_root(config)

    split_payload = {
        "train_indices": list(train_subset.indices),
        "test_indices": list(validation_subset.indices),
        "inference_indices": list(inference_subset.indices),
        "seed": int(config.get("seed", 42)),
        "total_size": total_size,
        "train_len": train_len,
        "test_len": test_len,
        "inference_len": inference_len,
    }
    torch.save(split_payload, split_manifest_path)

    variants = get_model_variants(config)
    summary_rows: List[Dict[str, Any]] = []
    for variant in variants:
        summary_rows.append(
            train_variant(
                config=config,
                variant=variant,
                train_subset=train_subset,
                validation_subset=validation_subset,
                train_label_index_pools=train_label_index_pools,
                validation_label_index_pools=validation_label_index_pools,
                device=device,
                used_config_path=used_config_path,
            )
        )

    manifest_payload = {
        "seed": int(config.get("seed", 42)),
        "device": str(device),
        "split_manifest": str(split_manifest_path),
        "variants": summary_rows,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    print(f"Saved split manifest to: {split_manifest_path}")
    print(f"Saved model-selection manifest to: {manifest_path}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Train configurable MNIST model variants for the SPLL pipeline.")
    parser.add_argument("--config", required=True, help="Path to the shared YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
