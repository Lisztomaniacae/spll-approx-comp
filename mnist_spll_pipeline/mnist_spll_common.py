"""Minimal compatibility surface required by the unchanged diagnostic utility.

Production stages import the focused modules directly. Keep this facade small:
its only purpose is to preserve the historical imports used by
``diagnose_cutoff_zero_mwe.py``.
"""

from mnist_model import resolve_device as _resolve_device
from pipeline1_models import (
    get_model_variants as _get_model_variants,
    get_variant_model_output_path as _get_variant_model_output_path,
)
from pipeline_support import load_config as _load_config


get_model_variants = _get_model_variants
get_variant_model_output_path = _get_variant_model_output_path
load_config = _load_config
resolve_device = _resolve_device

__all__ = [
    "get_model_variants",
    "get_variant_model_output_path",
    "load_config",
    "resolve_device",
]
