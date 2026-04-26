from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from mnist_spll_common import TerminalProgressBar
from mnist_spll_pipeline_core import (
    PipelinePaths,
    compiled_program_path,
    evaluate_candidate_sum,
    posterior_for_experiment,
    threshold_label,
    utc_now_iso,
)


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
    ) -> None:
        self.paths = paths
        self.model = model
        self.get_compiled_module = get_compiled_module
        self.show_progress = bool(show_progress)
        self.show_inner_progress = bool(show_inner_progress)
        self.progress_bar = progress_bar

    def run_many(
        self,
        *,
        experiments: Sequence[Dict[str, Any]],
        cutoff_modes: Iterable[str],
        thresholds: Iterable[Optional[float]],
    ) -> List[Dict[str, Any]]:
        raw_runs: List[Dict[str, Any]] = []
        for cutoff_mode in cutoff_modes:
            for experiment in experiments:
                for cutoff in thresholds:
                    raw_runs.append(self.run_one(experiment=experiment, cutoff_mode=cutoff_mode, cutoff=cutoff))
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
        started = time.perf_counter()
        posterior_trace = posterior_for_experiment(
            module,
            image_paths,
            max_sum=max_sum,
            progress_bar=per_run_bar,
            progress_prefix=(
                f"model={self.model.model_id}, cutoff_mode={cutoff_mode}, "
                f"exp={int(experiment['experiment_id']):04d}, terms={n_terms}, cutoff={label},"
            ),
        )
        runtime_sec = time.perf_counter() - started
        finished_at = utc_now_iso()

        true_candidate_trace = CandidateTrace.evaluate(module, image_paths, int(experiment["true_sum"]))

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
    ) -> Dict[str, Any]:
        compiled_path = compiled_program_path(self.paths.compiled_root, n_terms, cutoff_mode, cutoff)
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
        }
