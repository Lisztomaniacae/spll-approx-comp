from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset

from pipeline_support import resolve_path


class CNNClassifier(nn.Module):
    """Configurable MNIST classifier shared by both experiment pipelines."""

    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        super().__init__()
        input_channels = int(model_cfg.get("input_channels", 1))
        conv_channels = list(model_cfg.get("conv_channels", [32, 64]))
        kernel_size = int(model_cfg.get("kernel_size", 3))
        pool_kernel = int(model_cfg.get("pool_kernel", 2))
        fc_hidden = int(model_cfg.get("fc_hidden", 128))
        dropout = float(model_cfg.get("dropout", 0.25))
        num_classes = int(model_cfg.get("num_classes", 10))

        feature_layers = []
        in_channels = input_channels
        for out_channels in conv_channels:
            feature_layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(pool_kernel),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*feature_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 28, 28)
            feature_dim = int(np.prod(self.features(dummy).shape[1:]))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str = "auto", require_mps: bool = False) -> torch.device:
    normalized_name = device_name.lower()
    if normalized_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if require_mps:
            raise RuntimeError("MPS was required by config, but torch.backends.mps.is_available() is False.")
        return torch.device("cpu")
    if normalized_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Config requested device='mps', but MPS is not available.")
        return torch.device("mps")
    if normalized_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested device='cuda', but CUDA is not available.")
        return torch.device("cuda")
    if normalized_name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device setting: {device_name}")


def build_mnist_transform(config: Dict[str, Any]):
    from torchvision import transforms

    normalize_cfg = config["training"].get("normalize", {"mean": 0.1307, "std": 0.3081})
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (float(normalize_cfg["mean"]),),
                (float(normalize_cfg["std"]),),
            ),
        ]
    )


def load_full_mnist_transformed(config: Dict[str, Any]) -> ConcatDataset:
    from torchvision import datasets

    data_root = resolve_path(config, config["paths"]["data_root"])
    transform = build_mnist_transform(config)
    train_dataset = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transform)
    return ConcatDataset([train_dataset, test_dataset])


def load_full_mnist_raw(config: Dict[str, Any]) -> ConcatDataset:
    from torchvision import datasets

    data_root = resolve_path(config, config["paths"]["data_root"])
    train_dataset = datasets.MNIST(root=str(data_root), train=True, download=True, transform=None)
    test_dataset = datasets.MNIST(root=str(data_root), train=False, download=True, transform=None)
    return ConcatDataset([train_dataset, test_dataset])


def compute_split_lengths(
    total_size: int,
    train_ratio: float,
    test_ratio: float,
    inference_ratio: float,
) -> Tuple[int, int, int]:
    ratio_sum = train_ratio + test_ratio + inference_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}.")

    train_length = int(total_size * train_ratio)
    test_length = int(total_size * test_ratio)
    inference_length = total_size - train_length - test_length
    if min(train_length, test_length, inference_length) <= 0:
        raise ValueError(
            "Computed split sizes must all be positive, got "
            f"{(train_length, test_length, inference_length)} for total_size={total_size}."
        )
    return train_length, test_length, inference_length


def merge_model_config(
    base_model_cfg: Dict[str, Any],
    override_model_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged = copy.deepcopy(base_model_cfg)
    if override_model_cfg:
        merged.update(copy.deepcopy(override_model_cfg))
    return merged


def build_model(config: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None) -> CNNClassifier:
    final_model_cfg = merge_model_config(config["training"].get("model", {}), model_cfg)
    return CNNClassifier(final_model_cfg)


def load_checkpoint_model(
    checkpoint_path: str | Path,
    config: Dict[str, Any],
    map_location: str | torch.device = "cpu",
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_cfg = checkpoint.get("model_config", config["training"].get("model", {}))
    model = CNNClassifier(model_cfg)
    model.load_state_dict(checkpoint["state_dict"])
    return model
