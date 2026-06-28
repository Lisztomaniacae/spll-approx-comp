"""Minimal compatibility surface required by the unchanged diagnostic utility.

Pipeline I production code is split across focused modules. This facade keeps
only the historical imports still used by ``diagnose_cutoff_zero_mwe.py``.
"""

from pipeline1_config import build_pipeline_context as _build_pipeline_context
from pipeline1_data import (
    build_read_mnist as _build_read_mnist,
    load_staged_experiments as _load_staged_experiments,
)


build_pipeline_context = _build_pipeline_context
build_read_mnist = _build_read_mnist
load_staged_experiments = _load_staged_experiments

__all__ = [
    "build_pipeline_context",
    "build_read_mnist",
    "load_staged_experiments",
]
