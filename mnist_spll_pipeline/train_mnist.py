from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, random_split

from mnist_spll_common import (
    build_model,
    checkpoint_payload,
    compute_split_lengths,
    default_rng,
    ensure_dir,
    load_config,
    load_full_mnist_transformed,
    resolve_device,
    resolve_path,
    save_config,
    set_seed,
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a configurable MNIST CNN for the SPLL pipeline.")
    parser.add_argument("--config", required=True, help="Path to the shared YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
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
    train_subset, test_subset, inference_subset = random_split(
        full_dataset,
        [train_len, test_len, inference_len],
        generator=generator,
    )

    batch_size = int(training_cfg.get("batch_size", 128))
    eval_batch_size = int(training_cfg.get("eval_batch_size", 256))
    num_workers = int(training_cfg.get("num_workers", 0))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(config).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    export_path = resolve_path(config, paths_cfg["model_output"])
    split_manifest_path = resolve_path(config, paths_cfg["split_manifest"])
    metrics_path = resolve_path(config, paths_cfg.get("training_metrics_csv", "./outputs/training_metrics.csv"))
    used_config_path = resolve_path(config, paths_cfg.get("used_config_copy", "./outputs/config_used.yaml"))

    ensure_dir(export_path.parent)
    ensure_dir(split_manifest_path.parent)
    ensure_dir(metrics_path.parent)
    ensure_dir(used_config_path.parent)

    best_state = None
    best_epoch = -1
    best_test_accuracy = float("-inf")
    history: List[Dict[str, float]] = []

    epochs = int(training_cfg.get("epochs", 8))
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        test_metrics = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4%} | "
            f"test_loss={row['test_loss']:.4f} test_acc={row['test_accuracy']:.4%}"
        )

        if test_metrics["accuracy"] > best_test_accuracy:
            best_test_accuracy = float(test_metrics["accuracy"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    threshold = float(training_cfg.get("accuracy_threshold", 0.95))
    if best_state is None:
        raise RuntimeError("Training did not produce any model state.")

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    split_payload = {
        "train_indices": list(train_subset.indices),
        "test_indices": list(test_subset.indices),
        "inference_indices": list(inference_subset.indices),
        "seed": int(config.get("seed", 42)),
        "total_size": total_size,
        "train_len": train_len,
        "test_len": test_len,
        "inference_len": inference_len,
    }
    torch.save(split_payload, split_manifest_path)

    if best_test_accuracy < threshold:
        raise RuntimeError(
            f"Best test accuracy {best_test_accuracy:.4%} did not reach the configured threshold {threshold:.4%}. "
            f"Model was not exported to {export_path}."
        )

    model.load_state_dict(best_state)
    checkpoint = checkpoint_payload(
        model=model,
        config=config,
        best_epoch=best_epoch,
        best_test_accuracy=best_test_accuracy,
    )
    torch.save(checkpoint, export_path)
    save_config(config, used_config_path)

    print(f"Saved model to: {export_path}")
    print(f"Saved split manifest to: {split_manifest_path}")
    print(f"Saved metrics CSV to: {metrics_path}")
    print(
        f"Success: best test accuracy {best_test_accuracy:.4%} at epoch {best_epoch} exceeded the threshold {threshold:.4%}."
    )


if __name__ == "__main__":
    main()
