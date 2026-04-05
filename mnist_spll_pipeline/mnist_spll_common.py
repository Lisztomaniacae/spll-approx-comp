from __future__ import annotations

import copy
import hashlib
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
import yaml


class TerminalProgressBar:
    def __init__(
        self,
        total: int,
        *,
        desc: str = "Progress",
        unit: str = "items",
        enabled: bool = True,
        width: int = 28,
    ) -> None:
        self.total = max(int(total), 0)
        self.desc = desc
        self.unit = unit
        self.enabled = enabled
        self.width = max(int(width), 10)
        self.current = 0
        self.started_at = time.perf_counter()
        self._last_line_len = 0
        if self.enabled:
            self._render(force=True)

    def update(self, step: int = 1, *, postfix: str = "") -> None:
        self.current = min(self.total, self.current + int(step))
        if self.enabled:
            self._render(postfix=postfix)

    def finish(self, *, postfix: str = "done") -> None:
        self.current = self.total
        if self.enabled:
            self._render(postfix=postfix, force=True)
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _render(self, *, postfix: str = "", force: bool = False) -> None:
        if not self.enabled:
            return
        columns = shutil.get_terminal_size((100, 20)).columns
        usable_width = min(self.width, max(10, columns // 4))
        if self.total > 0:
            frac = min(max(self.current / self.total, 0.0), 1.0)
        else:
            frac = 1.0
        filled = int(round(usable_width * frac))
        bar = "#" * filled + "-" * (usable_width - filled)
        elapsed = time.perf_counter() - self.started_at
        rate = self.current / elapsed if elapsed > 0 and self.current > 0 else 0.0
        if rate > 0 and self.current < self.total:
            remaining = (self.total - self.current) / rate
            eta_text = f" ETA {remaining:5.1f}s"
        else:
            eta_text = ""
        line = (
            f"\r{self.desc}: [{bar}] {self.current}/{self.total} "
            f"({frac * 100:5.1f}%) {self.unit}{eta_text}"
        )
        if postfix:
            line += f" | {postfix}"
        pad = max(0, self._last_line_len - len(line))
        sys.stdout.write(line + (" " * pad))
        sys.stdout.flush()
        self._last_line_len = len(line)


def stage_message(current: int, total: int, message: str) -> None:
    print(f"\n[{current}/{total}] {message}", flush=True)


class CNNClassifier(nn.Module):
    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        super().__init__()
        input_channels = int(model_cfg.get("input_channels", 1))
        conv_channels = list(model_cfg.get("conv_channels", [32, 64]))
        kernel_size = int(model_cfg.get("kernel_size", 3))
        pool_kernel = int(model_cfg.get("pool_kernel", 2))
        fc_hidden = int(model_cfg.get("fc_hidden", 128))
        dropout = float(model_cfg.get("dropout", 0.25))
        num_classes = int(model_cfg.get("num_classes", 10))

        layers: List[nn.Module] = []
        in_ch = input_channels
        for out_ch in conv_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(pool_kernel))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Top-level YAML config must be a mapping.")
    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    return cfg


def save_config(config: Dict[str, Any], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(config)
    payload.pop("_config_path", None)
    payload.pop("_config_dir", None)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def resolve_path(config: Dict[str, Any], raw_path: str | Path) -> Path:
    raw_path = Path(raw_path)
    if raw_path.is_absolute():
        return raw_path
    return Path(config["_config_dir"]).joinpath(raw_path).resolve()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str = "auto", require_mps: bool = False) -> torch.device:
    device_name = device_name.lower()
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if require_mps:
            raise RuntimeError("MPS was required by config, but torch.backends.mps.is_available() is False.")
        return torch.device("cpu")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Config requested device='mps', but MPS is not available.")
        return torch.device("mps")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested device='cuda', but CUDA is not available.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device setting: {device_name}")


def build_train_transform(config: Dict[str, Any]):
    from torchvision import transforms

    normalize_cfg = config["training"].get("normalize", {"mean": 0.1307, "std": 0.3081})
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((float(normalize_cfg["mean"]),), (float(normalize_cfg["std"]),)),
        ]
    )


def build_eval_transform(config: Dict[str, Any]):
    return build_train_transform(config)


def load_full_mnist_transformed(config: Dict[str, Any], train: bool = True) -> ConcatDataset:
    from torchvision import datasets

    data_root = resolve_path(config, config["paths"]["data_root"])
    transform = build_train_transform(config) if train else build_eval_transform(config)
    ds_train = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
    ds_test = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transform)
    return ConcatDataset([ds_train, ds_test])


def load_full_mnist_raw(config: Dict[str, Any]) -> ConcatDataset:
    from torchvision import datasets

    data_root = resolve_path(config, config["paths"]["data_root"])
    ds_train = datasets.MNIST(root=str(data_root), train=True, download=True, transform=None)
    ds_test = datasets.MNIST(root=str(data_root), train=False, download=True, transform=None)
    return ConcatDataset([ds_train, ds_test])


def compute_split_lengths(total_size: int, train_ratio: float, test_ratio: float, inference_ratio: float) -> Tuple[int, int, int]:
    ratio_sum = train_ratio + test_ratio + inference_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}.")
    train_len = int(total_size * train_ratio)
    test_len = int(total_size * test_ratio)
    inference_len = total_size - train_len - test_len
    if min(train_len, test_len, inference_len) <= 0:
        raise ValueError(
            f"Computed split sizes must all be positive, got {(train_len, test_len, inference_len)} for total_size={total_size}."
        )
    return train_len, test_len, inference_len


def merge_model_config(base_model_cfg: Dict[str, Any], override_model_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = copy.deepcopy(base_model_cfg)
    if override_model_cfg:
        merged.update(copy.deepcopy(override_model_cfg))
    return merged


def build_model(config: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None) -> CNNClassifier:
    final_model_cfg = merge_model_config(config["training"].get("model", {}), model_cfg)
    return CNNClassifier(final_model_cfg)


def stable_variant_offset(variant_id: str) -> int:
    digest = hashlib.sha1(str(variant_id).encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def get_model_variants(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    training_cfg = config["training"]
    raw_variants = training_cfg.get("model_variants")
    base_model_cfg = copy.deepcopy(training_cfg.get("model", {}))

    if not raw_variants:
        default_target = training_cfg.get("target_accuracy", training_cfg.get("accuracy_threshold", 0.95))
        return [
            {
                "id": "default",
                "target_accuracy": float(default_target),
                "model": base_model_cfg,
                "epochs": int(training_cfg.get("epochs", 8)),
                "selection_mode": "nearest",
            }
        ]

    if not isinstance(raw_variants, list):
        raise ValueError("training.model_variants must be a list of mappings.")

    variants: List[Dict[str, Any]] = []
    seen_ids = set()
    for raw_variant in raw_variants:
        if not isinstance(raw_variant, dict):
            raise ValueError("Each training.model_variants entry must be a mapping.")
        variant_id = str(raw_variant.get("id", "")).strip()
        if not variant_id:
            raise ValueError("Each training.model_variants entry must define a non-empty 'id'.")
        if variant_id in seen_ids:
            raise ValueError(f"Duplicate model variant id: {variant_id}")
        seen_ids.add(variant_id)
        if "target_accuracy" not in raw_variant:
            raise ValueError(f"Model variant '{variant_id}' is missing required field 'target_accuracy'.")
        target_accuracy = float(raw_variant["target_accuracy"])
        if not (0.0 <= target_accuracy <= 1.0):
            raise ValueError(f"Variant '{variant_id}' target_accuracy must be in [0, 1], got {target_accuracy}.")
        selection_mode = str(raw_variant.get("selection_mode", training_cfg.get("selection_mode", "nearest"))).lower()
        if selection_mode not in {"nearest"}:
            raise ValueError(
                f"Variant '{variant_id}' selection_mode must currently be 'nearest', got {selection_mode}."
            )
        variant = copy.deepcopy(raw_variant)
        variant["id"] = variant_id
        variant["target_accuracy"] = target_accuracy
        variant["epochs"] = int(raw_variant.get("epochs", training_cfg.get("epochs", 8)))
        variant["selection_mode"] = selection_mode
        variant["model"] = merge_model_config(base_model_cfg, raw_variant.get("model", {}))
        variants.append(variant)
    return variants


def get_models_root(config: Dict[str, Any]) -> Path:
    paths_cfg = config["paths"]
    raw = paths_cfg.get("models_root")
    if raw is None:
        raw_model_output = paths_cfg.get("model_output")
        if raw_model_output is None:
            raw = "./outputs/models"
        else:
            raw = str(Path(raw_model_output).parent)
    return ensure_dir(resolve_path(config, raw))


def get_training_root(config: Dict[str, Any]) -> Path:
    paths_cfg = config["paths"]
    raw = paths_cfg.get("training_root")
    if raw is None:
        raw_metrics_output = paths_cfg.get("training_metrics_csv")
        if raw_metrics_output is None:
            raw = "./outputs/training"
        else:
            raw = str(Path(raw_metrics_output).parent)
    return ensure_dir(resolve_path(config, raw))


def get_variant_model_output_path(config: Dict[str, Any], variant_id: str) -> Path:
    return get_models_root(config) / f"{variant_id}.pt"


def get_variant_metrics_output_path(config: Dict[str, Any], variant_id: str) -> Path:
    return get_training_root(config) / f"{variant_id}_training_metrics.csv"


def get_model_selection_manifest_path(config: Dict[str, Any]) -> Path:
    paths_cfg = config["paths"]
    raw = paths_cfg.get("model_selection_manifest")
    if raw is None:
        return get_models_root(config) / "model_selection_manifest.json"
    return resolve_path(config, raw)


def checkpoint_payload(
    *,
    model: nn.Module,
    config: Dict[str, Any],
    best_epoch: int,
    best_test_accuracy: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "state_dict": model.state_dict(),
        "model_config": copy.deepcopy(config["training"]["model"]),
        "best_epoch": int(best_epoch),
        "best_test_accuracy": float(best_test_accuracy),
        "seed": int(config.get("seed", 42)),
    }
    if extra:
        payload.update(copy.deepcopy(extra))
    return payload


def load_checkpoint_model(checkpoint_path: str | Path, config: Dict[str, Any], map_location: str | torch.device = "cpu") -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_cfg = checkpoint.get("model_config", config["training"].get("model", {}))
    model = CNNClassifier(model_cfg)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def default_rng(config: Dict[str, Any], offset: int = 0) -> np.random.Generator:
    seed = int(config.get("seed", 42)) + int(offset)
    return np.random.default_rng(seed)
