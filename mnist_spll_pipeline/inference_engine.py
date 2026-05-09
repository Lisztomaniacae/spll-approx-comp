from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set

from mnist_spll_common import TerminalProgressBar
from mnist_spll_pipeline_core import (
    PipelinePaths,
    compiled_program_path,
    evaluate_candidate_sum,
    posterior_for_experiment,
    threshold_label,
    utc_now_iso,
)


READ_MNIST_CACHE_POLICY_RUN_SCOPED = "run_scoped_no_cross_run_cache"
READ_MNIST_CACHE_POLICY_UNCACHED = "uncached"
READ_MNIST_CACHE_POLICY_PRECOMPUTED = "precomputed_per_measurement"
READ_MNIST_CACHE_POLICY_DEFAULT = READ_MNIST_CACHE_POLICY_UNCACHED


def normalize_read_mnist_cache_policy(value: Any) -> str:
    """Return the canonical inference-time readMNist cache policy.

    Accepted aliases are intentionally lenient so old local configs can be
    updated without breaking immediately. The canonical values written to raw
    result metadata are:

    - ``run_scoped_no_cross_run_cache``: cache repeated image probabilities
      only inside one measured generated-SPLL query.
    - ``precomputed_per_measurement``: compute the probabilities for the
      current measurement's image paths immediately before timing, install a
      lookup-only ``readMNist`` for the timed generated-SPLL query, and discard
      it immediately afterwards.  This isolates symbolic/probabilistic
      inference time without allowing exact runs to warm approximate runs.
    - ``uncached``: call the base MNIST model for every generated-SPLL
      ``readMNist`` invocation. This is the default thesis timing policy.
    """

    text = str(value or READ_MNIST_CACHE_POLICY_DEFAULT).strip().lower().replace("-", "_")
    if text in {
        "cached",
        "run_scoped",
        "run_scoped_cache",
        "run_scoped_no_cross_run_cache",
    }:
        return READ_MNIST_CACHE_POLICY_RUN_SCOPED
    if text in {
        "precomputed",
        "precompute",
        "precomputed_lookup",
        "precomputed_per_measurement",
        "lookup_only",
    }:
        return READ_MNIST_CACHE_POLICY_PRECOMPUTED
    if text in {"uncached", "no_cache", "none", "off", "disabled", "false"}:
        return READ_MNIST_CACHE_POLICY_UNCACHED
    raise ValueError(
        "inference.read_mnist_cache_policy must be one of "
        f"'{READ_MNIST_CACHE_POLICY_UNCACHED}', '{READ_MNIST_CACHE_POLICY_RUN_SCOPED}', "
        f"or '{READ_MNIST_CACHE_POLICY_PRECOMPUTED}' "
        f"(got {value!r})."
    )


class RunScopedReadMNistCache:
    """Per-measurement readMNist cache with no state shared across runs.

    Generated SPLL code calls ``readMNist`` once per candidate sum, so caching the
    image probabilities inside one measured posterior query avoids timing the
    same neural forward pass again and again.  The important guardrail is scope:
    this cache is created fresh for exactly one timed measurement and discarded
    afterwards, so the exact baseline cannot warm the cache for later cutoff
    runs.
    """

    def __init__(self, base_read_mnist: Callable[[str], Sequence[float]]) -> None:
        self.base_read_mnist = base_read_mnist
        self.cache: Dict[str, List[float]] = {}
        self.calls = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def __call__(self, image_path: str) -> List[float]:
        self.calls += 1
        key = str(image_path)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return list(cached)

        self.cache_misses += 1
        probabilities = [float(value) for value in self.base_read_mnist(key)]
        self.cache[key] = probabilities
        return list(probabilities)

    def stats(self) -> Dict[str, int | str]:
        return {
            "policy": READ_MNIST_CACHE_POLICY_RUN_SCOPED,
            "calls": int(self.calls),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "unique_images": int(len(self.cache)),
        }


class UncachedReadMNistCounter:
    """Counting proxy that deliberately does not cache readMNist results."""

    def __init__(self, base_read_mnist: Callable[[str], Sequence[float]]) -> None:
        self.base_read_mnist = base_read_mnist
        self.calls = 0
        self.unique_images_seen: Set[str] = set()

    def __call__(self, image_path: str) -> List[float]:
        self.calls += 1
        key = str(image_path)
        self.unique_images_seen.add(key)
        return [float(value) for value in self.base_read_mnist(key)]

    def stats(self) -> Dict[str, int | str]:
        return {
            "policy": READ_MNIST_CACHE_POLICY_UNCACHED,
            "calls": int(self.calls),
            "cache_hits": 0,
            "cache_misses": int(self.calls),
            "unique_images": int(len(self.unique_images_seen)),
        }


class PrecomputedReadMNistLookup:
    """Lookup-only readMNist proxy with fresh per-measurement precompute.

    The precompute step intentionally happens inside the measurement scope setup
    but outside the timed posterior/true-candidate query.  A new instance is
    created for every exact/cutoff measurement, so no run can benefit from a
    cache warmed by a previous run.
    """

    def __init__(self, base_read_mnist: Callable[[str], Sequence[float]], image_paths: Sequence[str]) -> None:
        self.base_read_mnist = base_read_mnist
        self.calls = 0
        self.lookup_hits = 0
        self.lookup_misses = 0
        self.cache: Dict[str, List[float]] = {}

        unique_paths: List[str] = []
        seen: Set[str] = set()
        for image_path in image_paths:
            key = str(image_path)
            if key not in seen:
                seen.add(key)
                unique_paths.append(key)

        started = time.perf_counter()
        for key in unique_paths:
            self.cache[key] = [float(value) for value in self.base_read_mnist(key)]
        self.precompute_runtime_sec = float(time.perf_counter() - started)
        self.precompute_calls = int(len(unique_paths))

    def __call__(self, image_path: str) -> List[float]:
        self.calls += 1
        key = str(image_path)
        cached = self.cache.get(key)
        if cached is None:
            self.lookup_misses += 1
            raise KeyError(
                "precomputed_per_measurement readMNist received an image path "
                f"that was not precomputed for this measurement: {key!r}"
            )
        self.lookup_hits += 1
        return list(cached)

    def stats(self) -> Dict[str, int | float | str]:
        return {
            "policy": READ_MNIST_CACHE_POLICY_PRECOMPUTED,
            "calls": int(self.calls),
            "cache_hits": int(self.lookup_hits),
            "cache_misses": int(self.lookup_misses),
            "unique_images": int(len(self.cache)),
            "precompute_calls": int(self.precompute_calls),
            "precompute_runtime_sec": float(self.precompute_runtime_sec),
        }


def warm_up_read_mnist(
    read_mnist: Callable[[str], Sequence[float]],
    experiments: Sequence[Dict[str, Any]],
    calls: int,
) -> Dict[str, Any]:
    """Run untimed readMNist calls before measured inference.

    This removes one-off PIL/PyTorch/device-dispatch startup effects from the
    first exact/cutoff measurement without installing any prediction cache.
    The callback is intentionally the same uncached base callback that compiled
    SPLL modules will call later through the selected cache policy.
    """

    requested = max(0, int(calls))
    if requested <= 0 or not experiments:
        return {"calls": 0, "runtime_sec": 0.0, "image_paths": []}

    candidate_paths: List[str] = []
    for experiment in experiments:
        for image_path in experiment.get("image_paths", []):
            candidate_paths.append(str(image_path))
            if len(candidate_paths) >= requested:
                break
        if len(candidate_paths) >= requested:
            break

    if not candidate_paths:
        return {"calls": 0, "runtime_sec": 0.0, "image_paths": []}

    started = time.perf_counter()
    completed = 0
    for index in range(requested):
        image_path = candidate_paths[index % len(candidate_paths)]
        read_mnist(image_path)
        completed += 1
    runtime_sec = time.perf_counter() - started
    return {
        "calls": int(completed),
        "runtime_sec": float(runtime_sec),
        "image_paths": candidate_paths,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelInferenceContext:
    """Stable model metadata copied into every raw inference run record."""

    model_id: str
    target_accuracy: float
    selected_epoch: int
    selected_test_accuracy: float
    model_path: Path

    @classmethod
    def from_checkpoint(
        cls,
        *,
        model_id: str,
        target_accuracy: float,
        model_path: Path,
        checkpoint_meta: Dict[str, Any],
    ) -> "ModelInferenceContext":
        return cls(
            model_id=model_id,
            target_accuracy=float(target_accuracy),
            selected_epoch=int(checkpoint_meta.get("selected_epoch", checkpoint_meta.get("best_epoch", -1))),
            selected_test_accuracy=float(
                checkpoint_meta.get(
                    "selected_test_accuracy",
                    checkpoint_meta.get("best_test_accuracy", 0.0),
                )
            ),
            model_path=Path(model_path),
        )


@dataclass(frozen=True)
class CandidateTrace:
    """Result of one compiled SPLL query for a single candidate sum."""

    candidate_sum: int
    probability_raw: float
    branch_count: Optional[int]
    runtime_sec: float
    started_at_utc: str
    finished_at_utc: str

    @classmethod
    def evaluate(cls, module: Any, image_paths: Sequence[str], candidate_sum: int) -> "CandidateTrace":
        started_at = utc_now_iso()
        started = time.perf_counter()
        trace = evaluate_candidate_sum(module, image_paths, int(candidate_sum))
        runtime_sec = time.perf_counter() - started
        finished_at = utc_now_iso()
        branch_count = trace["branch_count"]
        return cls(
            candidate_sum=int(trace["candidate_sum"]),
            probability_raw=float(trace["probability_raw"]),
            branch_count=None if branch_count is None else int(branch_count),
            runtime_sec=float(runtime_sec),
            started_at_utc=started_at,
            finished_at_utc=finished_at,
        )


class InferenceRunEngine:
    """
    Owns the execution and raw-result schema for MNIST SPLL inference runs.

    Stage orchestration decides which models/experiments/cutoffs exist. This
    module owns the repeated boundary behavior: load the compiled module, query
    the whole posterior, query the true candidate separately, time both, and emit
    the stable raw JSON row consumed by visualization.
    """

    def __init__(
        self,
        *,
        paths: PipelinePaths,
        model: ModelInferenceContext,
        get_compiled_module: Callable[[int, str, Optional[float]], Any],
        show_progress: bool,
        show_inner_progress: bool,
        progress_bar: Optional[TerminalProgressBar] = None,
        read_mnist_cache_policy: str = READ_MNIST_CACHE_POLICY_DEFAULT,
        read_mnist_warmup_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.paths = paths
        self.model = model
        self.get_compiled_module = get_compiled_module
        self.show_progress = bool(show_progress)
        self.show_inner_progress = bool(show_inner_progress)
        self.progress_bar = progress_bar
        self.read_mnist_cache_policy = normalize_read_mnist_cache_policy(read_mnist_cache_policy)
        self.read_mnist_warmup_stats = dict(read_mnist_warmup_stats or {})

    @contextmanager
    def _read_mnist_scope(
        self,
        module: Any,
        *,
        image_paths: Sequence[str] = (),
    ) -> Iterator[RunScopedReadMNistCache | UncachedReadMNistCounter | PrecomputedReadMNistLookup]:
        base_read_mnist = getattr(module, "_spll_base_readMNist", getattr(module, "readMNist"))
        previous_read_mnist = getattr(module, "readMNist")
        if self.read_mnist_cache_policy == READ_MNIST_CACHE_POLICY_RUN_SCOPED:
            scoped_read_mnist: RunScopedReadMNistCache | UncachedReadMNistCounter | PrecomputedReadMNistLookup = RunScopedReadMNistCache(base_read_mnist)
        elif self.read_mnist_cache_policy == READ_MNIST_CACHE_POLICY_UNCACHED:
            scoped_read_mnist = UncachedReadMNistCounter(base_read_mnist)
        elif self.read_mnist_cache_policy == READ_MNIST_CACHE_POLICY_PRECOMPUTED:
            scoped_read_mnist = PrecomputedReadMNistLookup(base_read_mnist, image_paths)
        else:
            raise ValueError(f"Unsupported readMNist cache policy: {self.read_mnist_cache_policy!r}")
        setattr(module, "readMNist", scoped_read_mnist)
        try:
            yield scoped_read_mnist
        finally:
            setattr(module, "readMNist", previous_read_mnist)

    @staticmethod
    def _rotated_thresholds(thresholds: Sequence[Optional[float]], offset: int) -> List[Optional[float]]:
        values = list(thresholds)
        if not values:
            return []
        shift = int(offset) % len(values)
        return values[shift:] + values[:shift]

    def run_many(
        self,
        *,
        experiments: Sequence[Dict[str, Any]],
        cutoff_modes: Iterable[str],
        thresholds: Iterable[Optional[float]],
    ) -> List[Dict[str, Any]]:
        raw_runs: List[Dict[str, Any]] = []
        threshold_values = list(thresholds)
        measurement_order_index = 0
        for cutoff_mode in cutoff_modes:
            for experiment_index, experiment in enumerate(experiments):
                for threshold_position, cutoff in enumerate(self._rotated_thresholds(threshold_values, experiment_index)):
                    record = self.run_one(experiment=experiment, cutoff_mode=cutoff_mode, cutoff=cutoff)
                    record["measurement_order_index"] = int(measurement_order_index)
                    record["threshold_order_position"] = int(threshold_position)
                    raw_runs.append(record)
                    measurement_order_index += 1
        return raw_runs

    def run_one(self, *, experiment: Dict[str, Any], cutoff_mode: str, cutoff: Optional[float]) -> Dict[str, Any]:
        n_terms = int(experiment["n_terms"])
        image_paths = [str(path) for path in experiment["image_paths"]]
        max_sum = 9 * n_terms
        label = threshold_label(cutoff)
        module = self.get_compiled_module(n_terms, cutoff_mode, cutoff)

        per_run_bar = TerminalProgressBar(
            max_sum + 1,
            desc="  Posterior",
            unit="sums",
            enabled=self.show_progress and self.show_inner_progress and (max_sum + 1) > 0,
        )
        started_at = utc_now_iso()
        with self._read_mnist_scope(module, image_paths=image_paths) as posterior_read_mnist:
            started = time.perf_counter()
            posterior_trace = posterior_for_experiment(
                module,
                image_paths,
                max_sum=max_sum,
                progress_bar=None,
                progress_prefix=(
                    f"model={self.model.model_id}, cutoff_mode={cutoff_mode}, "
                    f"exp={int(experiment['experiment_id']):04d}, terms={n_terms}, cutoff={label},"
                ),
            )
            runtime_sec = time.perf_counter() - started
            posterior_read_mnist_stats = posterior_read_mnist.stats()
        finished_at = utc_now_iso()

        with self._read_mnist_scope(module, image_paths=image_paths) as true_candidate_read_mnist:
            true_candidate_trace = CandidateTrace.evaluate(module, image_paths, int(experiment["true_sum"]))
            true_candidate_read_mnist_stats = true_candidate_read_mnist.stats()

        per_run_bar.finish(
            postfix=(
                f"model={self.model.model_id}, cutoff_mode={cutoff_mode}, "
                f"exp={int(experiment['experiment_id']):04d}, terms={n_terms}, cutoff={label}, "
                f"runtime={runtime_sec:.2f}s, true_sum_runtime={true_candidate_trace.runtime_sec:.4f}s"
            )
        )

        record = self._build_record(
            experiment=experiment,
            cutoff_mode=cutoff_mode,
            cutoff=cutoff,
            label=label,
            n_terms=n_terms,
            image_paths=image_paths,
            max_sum=max_sum,
            posterior_trace=posterior_trace,
            runtime_sec=runtime_sec,
            started_at=started_at,
            finished_at=finished_at,
            true_candidate_trace=true_candidate_trace,
            posterior_read_mnist_stats=posterior_read_mnist_stats,
            true_candidate_read_mnist_stats=true_candidate_read_mnist_stats,
        )
        if self.progress_bar is not None:
            self.progress_bar.update(
                postfix=(
                    f"model={self.model.model_id}, cutoff_mode={cutoff_mode}, "
                    f"exp={int(experiment['experiment_id']):04d}, terms={n_terms}, cutoff={label}, "
                    f"runtime={runtime_sec:.2f}s"
                )
            )
        return record

    def _build_record(
        self,
        *,
        experiment: Dict[str, Any],
        cutoff_mode: str,
        cutoff: Optional[float],
        label: str,
        n_terms: int,
        image_paths: Sequence[str],
        max_sum: int,
        posterior_trace: Dict[str, Any],
        runtime_sec: float,
        started_at: str,
        finished_at: str,
        true_candidate_trace: CandidateTrace,
        posterior_read_mnist_stats: Dict[str, Any],
        true_candidate_read_mnist_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        compiled_path = compiled_program_path(self.paths.compiled_root, n_terms, cutoff_mode, cutoff)
        compiled_program_sha256 = sha256_file(compiled_path) if compiled_path.exists() else ""
        return {
            "model_id": self.model.model_id,
            "target_accuracy": self.model.target_accuracy,
            "selected_epoch": self.model.selected_epoch,
            "selected_test_accuracy": self.model.selected_test_accuracy,
            "experiment_id": int(experiment["experiment_id"]),
            "cutoff_mode": cutoff_mode,
            "n_terms": int(n_terms),
            "cutoff": cutoff,
            "threshold_label": label,
            "candidate_sums": list(range(max_sum + 1)),
            "posterior_raw": [float(value) for value in posterior_trace["posterior_raw"]],
            "branch_counts_raw": [
                None if value is None else int(value) for value in posterior_trace["branch_counts_raw"]
            ],
            "runtime_sec": float(runtime_sec),
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "true_candidate_sum": true_candidate_trace.candidate_sum,
            "true_candidate_probability_raw": true_candidate_trace.probability_raw,
            "true_candidate_branch_count": true_candidate_trace.branch_count,
            "true_candidate_runtime_sec": true_candidate_trace.runtime_sec,
            "true_candidate_started_at_utc": true_candidate_trace.started_at_utc,
            "true_candidate_finished_at_utc": true_candidate_trace.finished_at_utc,
            "true_sum": int(experiment["true_sum"]),
            "labels": [int(value) for value in experiment["labels"]],
            "global_indices": [int(value) for value in experiment["global_indices"]],
            "image_paths": list(image_paths),
            "model_path": str(self.model.model_path),
            "compiled_program_path": str(compiled_path),
            "compiled_program_sha256": compiled_program_sha256,
            "read_mnist_cache_policy": self.read_mnist_cache_policy,
            "posterior_read_mnist_stats": posterior_read_mnist_stats,
            "true_candidate_read_mnist_stats": true_candidate_read_mnist_stats,
            "read_mnist_precompute_runtime_sec": float(
                posterior_read_mnist_stats.get("precompute_runtime_sec", 0.0)
            ),
            "true_candidate_read_mnist_precompute_runtime_sec": float(
                true_candidate_read_mnist_stats.get("precompute_runtime_sec", 0.0)
            ),
            "read_mnist_precompute_stats": posterior_read_mnist_stats,
            "true_candidate_read_mnist_precompute_stats": true_candidate_read_mnist_stats,
            "read_mnist_warmup_calls": int(self.read_mnist_warmup_stats.get("calls", 0)),
            "read_mnist_warmup_runtime_sec": float(self.read_mnist_warmup_stats.get("runtime_sec", 0.0)),
        }
