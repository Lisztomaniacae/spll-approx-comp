from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mnist_model import merge_model_config
from pipeline_support import resolve_path


def stable_variant_offset(variant_id: str) -> int:
    digest = hashlib.sha1(str(variant_id).encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def get_model_variants(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    training_cfg = config["training"]
    raw_variants = training_cfg.get("model_variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("training.model_variants must be a non-empty list of mappings.")

    base_model_cfg = copy.deepcopy(training_cfg.get("model", {}))
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
        if not 0.0 <= target_accuracy <= 1.0:
            raise ValueError(
                f"Variant '{variant_id}' target_accuracy must be in [0, 1], got {target_accuracy}."
            )

        variant = copy.deepcopy(raw_variant)
        variant["id"] = variant_id
        variant["target_accuracy"] = target_accuracy
        variant["epochs"] = int(raw_variant.get("epochs", training_cfg.get("epochs", 8)))
        variant["selection_mode"] = "nearest"
        variant["visualization_group"] = (
            str(raw_variant.get("visualization_group", "main")).strip() or "main"
        )
        variant["model"] = merge_model_config(base_model_cfg, raw_variant.get("model", {}))
        variants.append(variant)

    return variants


def get_models_root(config: Dict[str, Any]) -> Path:
    return resolve_path(config, config["paths"].get("models_root", "./outputs/models"))


def get_training_root(config: Dict[str, Any]) -> Path:
    return resolve_path(config, config["paths"].get("training_root", "./outputs/training"))


def get_variant_model_output_path(config: Dict[str, Any], variant_id: str) -> Path:
    return get_models_root(config) / f"{variant_id}.pt"


def get_variant_metrics_output_path(config: Dict[str, Any], variant_id: str) -> Path:
    return get_training_root(config) / f"{variant_id}_training_metrics.csv"


def get_model_selection_manifest_path(config: Dict[str, Any]) -> Path:
    raw_path = config["paths"].get("model_selection_manifest")
    return (
        resolve_path(config, raw_path)
        if raw_path is not None
        else get_models_root(config) / "model_selection_manifest.json"
    )


def default_rng(config: Dict[str, Any], offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(int(config.get("seed", 42)) + int(offset))
