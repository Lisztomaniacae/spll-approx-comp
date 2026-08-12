from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from pipeline1_config import get_thresholds
from pipeline_support import load_json


EPS = 1e-12


def set_analysis_seed(seed: int) -> None:
    """Seed the random generators used by analysis without importing PyTorch."""

    random.seed(int(seed))
    np.random.seed(int(seed))


def normalize_distribution(values: Sequence[float]) -> List[float]:
    total = float(sum(values))
    if total <= 0:
        return [0.0 for _ in values]
    return [float(v) / total for v in values]


def entropy_from_distribution(values: Sequence[float]) -> float:
    positive = [float(v) for v in values if float(v) > 0.0]
    if not positive:
        return 0.0
    return float(-sum(v * math.log(v) for v in positive))


def top_predictions(posterior: Sequence[float], k: int) -> List[Dict[str, float]]:
    indexed = sorted(enumerate(posterior), key=lambda item: item[1], reverse=True)[:k]
    return [{"sum": int(idx), "probability": float(prob)} for idx, prob in indexed]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_payload_runs(path: Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        return payload["runs"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected inference run payload at {path}")


def load_payload_experiments(path: Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("experiments"), list):
        return payload["experiments"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected staged experiment payload at {path}")


def ordered_threshold_labels(config: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    for threshold in get_thresholds(config):
        labels.append(str(threshold["threshold_label"]))
    if "exact" in labels:
        labels = ["exact"] + [label for label in labels if label != "exact"]
    seen = set()
    ordered: List[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def non_exact_threshold_labels(threshold_order: Sequence[str]) -> List[str]:
    return [label for label in threshold_order if label != "exact"]


def pretty_threshold_label(label: str) -> str:
    if label == "exact":
        return "exact"
    if label.startswith("approx_mass_"):
        return "mass " + label.removeprefix("approx_mass_").replace("p", ".")
    if not label.startswith("cutoff_"):
        return label
    raw = label.removeprefix("cutoff_").replace("p", ".")
    try:
        value = float(raw)
    except ValueError:
        return label
    if value == 0.0:
        return "0.0"
    if value >= 0.1:
        return f"{value:.2g}"
    exponent = int(round(math.log10(value)))
    if abs(value - 10**exponent) < 1e-12:
        return f"1e{exponent}"
    mantissa = value / (10 ** math.floor(math.log10(value)))
    return f"{mantissa:g}e{math.floor(math.log10(value))}"


def threshold_sort_key(row: Dict[str, Any], threshold_order: Sequence[str]) -> Tuple[int, float]:
    label = str(row["threshold_label"])
    try:
        label_idx = threshold_order.index(label)
    except ValueError:
        label_idx = len(threshold_order)
    cutoff = row.get("cutoff")
    cutoff_value = -1.0 if cutoff is None else float(cutoff)
    return label_idx, cutoff_value


def compact_model_name(row: Dict[str, Any]) -> str:
    target_pct = int(round(100.0 * float(row.get("target_accuracy", row.get("selected_test_accuracy", 0.0)))))
    return f"{target_pct}%"


def model_label(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "model"
    return compact_model_name(rows[0])


def quantile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return float("nan")
    return float(np.quantile(array, q))


def finite_float_values(items: Iterable[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for item in items:
        try:
            value = float(item.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean_or_nan(values: Sequence[float]) -> float:
    return mean(values) if values else float("nan")


def median_or_nan(values: Sequence[float]) -> float:
    return median(values) if values else float("nan")


def read_mnist_count_from_run(
        run: Dict[str, Any],
        *,
        true_candidate: bool,
        model_evaluations: bool,
) -> float:
    """Read a scalar MNIST-call count with backward-compatible fallbacks.

    New inference runs expose flat scalar fields. Existing result bundles from
    the current pipeline already contain the same information inside the nested
    ``*_read_mnist_stats`` dictionaries, so visualization can be regenerated
    without rerunning inference.
    """

    prefix = "true_candidate_" if true_candidate else ""
    explicit_key = prefix + (
        "read_mnist_model_evaluations" if model_evaluations else "read_mnist_lookup_calls"
    )
    explicit = run.get(explicit_key)
    if explicit is not None:
        try:
            return float(explicit)
        except (TypeError, ValueError):
            pass

    stats_key = "true_candidate_read_mnist_stats" if true_candidate else "posterior_read_mnist_stats"
    stats = run.get(stats_key)
    if not isinstance(stats, dict):
        return float("nan")

    if not model_evaluations:
        value = stats.get("calls")
    elif str(stats.get("policy", run.get("read_mnist_cache_policy", ""))) == "precomputed_per_measurement":
        value = stats.get("precompute_calls")
    else:
        value = stats.get("cache_misses", stats.get("calls"))

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize_groups(
        rows: List[Dict[str, Any]],
        group_keys: Sequence[str],
        threshold_order: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    summary: List[Dict[str, Any]] = []
    for items in grouped.values():
        first = items[0]
        runtimes = [float(item["runtime_sec"]) for item in items]
        true_candidate_runtimes = finite_float_values(items, "true_candidate_runtime_sec")
        true_candidate_branch_counts = finite_float_values(items, "true_candidate_branch_count")
        precompute_runtimes = finite_float_values(items, "read_mnist_precompute_runtime_sec")
        read_mnist_lookup_calls = finite_float_values(items, "read_mnist_lookup_calls")
        read_mnist_model_evaluations = finite_float_values(items, "read_mnist_model_evaluations")
        runtime_top_k_cutoffs = finite_float_values(items, "runtime_top_k_cutoff")
        surviving_masses = finite_float_values(items, "mean_surviving_posterior_mass")
        adaptive_search_runtimes = finite_float_values(items, "adaptive_cutoff_search_runtime_sec")
        true_candidate_precompute_runtimes = finite_float_values(
            items,
            "true_candidate_read_mnist_precompute_runtime_sec",
        )
        true_candidate_read_mnist_lookup_calls = finite_float_values(
            items,
            "true_candidate_read_mnist_lookup_calls",
        )
        true_candidate_read_mnist_model_evaluations = finite_float_values(
            items,
            "true_candidate_read_mnist_model_evaluations",
        )
        result: Dict[str, Any] = {key: first[key] for key in group_keys}
        result.update(
            {
                "visualization_group": str(first.get("visualization_group", "main")),
                "target_accuracy": float(first.get("target_accuracy", 0.0)),
                "selected_epoch": int(first.get("selected_epoch", -1)),
                "selected_test_accuracy": float(first.get("selected_test_accuracy", 0.0)),
                "experiments": len(items),
                "adaptive_top_k": bool(first.get("adaptive_top_k", False)),
                "posterior_mass_target": first.get("posterior_mass_target"),
                "mean_runtime_top_k_cutoff": mean_or_nan(runtime_top_k_cutoffs),
                "median_runtime_top_k_cutoff": median_or_nan(runtime_top_k_cutoffs),
                "mean_surviving_posterior_mass": mean_or_nan(surviving_masses),
                "mean_adaptive_cutoff_search_runtime_sec": mean_or_nan(adaptive_search_runtimes),
                "accuracy": mean(float(item["correct"]) for item in items),
                "mean_runtime_sec": mean(runtimes),
                "median_runtime_sec": median(runtimes),
                "runtime_q25_sec": quantile(runtimes, 0.25),
                "runtime_q75_sec": quantile(runtimes, 0.75),
                "mean_read_mnist_precompute_runtime_sec": mean_or_nan(precompute_runtimes),
                "median_read_mnist_precompute_runtime_sec": median_or_nan(precompute_runtimes),
                "mean_read_mnist_lookup_calls": mean_or_nan(read_mnist_lookup_calls),
                "median_read_mnist_lookup_calls": median_or_nan(read_mnist_lookup_calls),
                "mean_read_mnist_model_evaluations": mean_or_nan(read_mnist_model_evaluations),
                "median_read_mnist_model_evaluations": median_or_nan(read_mnist_model_evaluations),
                "mean_confidence": mean(float(item["confidence"]) for item in items),
                "mean_output_pool": mean(float(item["output_pool"]) for item in items),
                "mean_output_pool_fraction": mean(float(item["output_pool_fraction"]) for item in items),
                "mean_total_branch_count": mean(float(item["total_branch_count"]) for item in items),
                "mean_max_branch_count": mean(float(item["max_branch_count"]) for item in items),
                "mean_posterior_entropy": mean(float(item["posterior_entropy"]) for item in items),
                "mean_posterior_mass": mean(float(item["posterior_mass"]) for item in items),
                "zero_mass_rate": mean(float(item["zero_mass"]) for item in items),
                "mean_candidate_count": mean(float(item["candidate_count"]) for item in items),
                "mean_true_candidate_runtime_sec": mean_or_nan(true_candidate_runtimes),
                "median_true_candidate_runtime_sec": median_or_nan(true_candidate_runtimes),
                "true_candidate_runtime_q25_sec": quantile(true_candidate_runtimes, 0.25),
                "true_candidate_runtime_q75_sec": quantile(true_candidate_runtimes, 0.75),
                "mean_true_candidate_read_mnist_precompute_runtime_sec": mean_or_nan(
                    true_candidate_precompute_runtimes
                ),
                "median_true_candidate_read_mnist_precompute_runtime_sec": median_or_nan(
                    true_candidate_precompute_runtimes
                ),
                "mean_true_candidate_read_mnist_lookup_calls": mean_or_nan(
                    true_candidate_read_mnist_lookup_calls
                ),
                "median_true_candidate_read_mnist_lookup_calls": median_or_nan(
                    true_candidate_read_mnist_lookup_calls
                ),
                "mean_true_candidate_read_mnist_model_evaluations": mean_or_nan(
                    true_candidate_read_mnist_model_evaluations
                ),
                "median_true_candidate_read_mnist_model_evaluations": median_or_nan(
                    true_candidate_read_mnist_model_evaluations
                ),
                "mean_true_candidate_probability_raw": mean(
                    float(item["true_candidate_probability_raw"]) for item in items
                ),
                "mean_true_candidate_normalized_probability": mean(
                    float(item["true_candidate_normalized_probability"]) for item in items
                ),
                "mean_true_candidate_branch_count": mean_or_nan(true_candidate_branch_counts),
                "true_candidate_survival_rate": mean(float(item["true_candidate_survived"]) for item in items),
                "mean_true_candidate_branch_fraction_of_total": mean_or_nan(
                    finite_float_values(items, "true_candidate_branch_fraction_of_total")
                ),
                "mean_true_candidate_runtime_fraction_of_full": mean_or_nan(
                    finite_float_values(items, "true_candidate_runtime_fraction_of_full")
                ),
            }
        )
        summary.append(result)

    summary.sort(
        key=lambda row: (
            row.get("model_id", ""),
            int(row.get("n_terms", -1)) if "n_terms" in row else -1,
            threshold_sort_key(row, threshold_order),
        )
    )
    return summary


def add_exact_baseline_columns(summary_rows: List[Dict[str, Any]], group_keys: Sequence[str]) -> None:
    keys_wo_threshold = [key for key in group_keys if key not in {"threshold_label", "cutoff"}]
    baseline_by_group: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in summary_rows:
        if row.get("threshold_label") == "exact":
            baseline_by_group[tuple(row[key] for key in keys_wo_threshold)] = row

    for row in summary_rows:
        baseline = baseline_by_group.get(tuple(row[key] for key in keys_wo_threshold))
        runtime = float(row["median_runtime_sec"])
        true_runtime = float(row.get("median_true_candidate_runtime_sec", float("nan")))
        if baseline is None:
            row["speedup_vs_exact"] = float("nan")
            row["runtime_ratio_vs_exact"] = float("nan")
            row["accuracy_delta_vs_exact"] = float("nan")
            row["confidence_delta_vs_exact"] = float("nan")
            row["output_pool_fraction_delta_vs_exact"] = float("nan")
            row["branch_count_delta_vs_exact"] = float("nan")
            row["collapse_rate_delta_vs_exact"] = float("nan")
            row["true_candidate_speedup_vs_exact"] = float("nan")
            row["true_candidate_runtime_ratio_vs_exact"] = float("nan")
            row["true_candidate_probability_delta_vs_exact"] = float("nan")
            row["true_candidate_normalized_probability_delta_vs_exact"] = float("nan")
            row["true_candidate_branch_count_delta_vs_exact"] = float("nan")
            row["true_candidate_survival_rate_delta_vs_exact"] = float("nan")
            row["read_mnist_lookup_ratio_vs_exact"] = float("nan")
            row["read_mnist_lookup_reduction_vs_exact"] = float("nan")
            row["read_mnist_lookup_speedup_vs_exact"] = float("nan")
            row["read_mnist_model_evaluation_ratio_vs_exact"] = float("nan")
            row["read_mnist_model_evaluation_reduction_vs_exact"] = float("nan")
            row["true_candidate_read_mnist_lookup_ratio_vs_exact"] = float("nan")
            row["true_candidate_read_mnist_lookup_reduction_vs_exact"] = float("nan")
            continue

        baseline_runtime = float(baseline["median_runtime_sec"])
        row["speedup_vs_exact"] = float(baseline_runtime / runtime) if runtime > 0 else float("nan")
        row["runtime_ratio_vs_exact"] = float(runtime / baseline_runtime) if baseline_runtime > 0 else float("nan")
        row["accuracy_delta_vs_exact"] = float(row["accuracy"]) - float(baseline["accuracy"])
        row["confidence_delta_vs_exact"] = float(row["mean_confidence"]) - float(baseline["mean_confidence"])
        row["output_pool_fraction_delta_vs_exact"] = float(row["mean_output_pool_fraction"]) - float(
            baseline["mean_output_pool_fraction"]
        )
        row["branch_count_delta_vs_exact"] = float(row["mean_total_branch_count"]) - float(
            baseline["mean_total_branch_count"]
        )
        row["collapse_rate_delta_vs_exact"] = float(row["zero_mass_rate"]) - float(baseline["zero_mass_rate"])

        baseline_true_runtime = float(baseline.get("median_true_candidate_runtime_sec", float("nan")))
        row["true_candidate_speedup_vs_exact"] = (
            float(baseline_true_runtime / true_runtime)
            if baseline_true_runtime > 0 and true_runtime > 0
            else float("nan")
        )
        row["true_candidate_runtime_ratio_vs_exact"] = (
            float(true_runtime / baseline_true_runtime)
            if baseline_true_runtime > 0 and true_runtime > 0
            else float("nan")
        )
        row["true_candidate_probability_delta_vs_exact"] = float(
            row["mean_true_candidate_probability_raw"]
        ) - float(baseline["mean_true_candidate_probability_raw"])
        row["true_candidate_normalized_probability_delta_vs_exact"] = float(
            row["mean_true_candidate_normalized_probability"]
        ) - float(baseline["mean_true_candidate_normalized_probability"])
        row["true_candidate_branch_count_delta_vs_exact"] = float(
            row["mean_true_candidate_branch_count"]
        ) - float(baseline["mean_true_candidate_branch_count"])
        row["true_candidate_survival_rate_delta_vs_exact"] = float(
            row["true_candidate_survival_rate"]
        ) - float(baseline["true_candidate_survival_rate"])

        current_lookups = float(row.get("mean_read_mnist_lookup_calls", float("nan")))
        baseline_lookups = float(baseline.get("mean_read_mnist_lookup_calls", float("nan")))
        lookup_ratio = (
            float(current_lookups / baseline_lookups)
            if math.isfinite(current_lookups) and math.isfinite(baseline_lookups) and baseline_lookups > 0
            else float("nan")
        )
        row["read_mnist_lookup_ratio_vs_exact"] = lookup_ratio
        row["read_mnist_lookup_reduction_vs_exact"] = (
            float(1.0 - lookup_ratio) if math.isfinite(lookup_ratio) else float("nan")
        )
        row["read_mnist_lookup_speedup_vs_exact"] = (
            float(baseline_lookups / current_lookups)
            if math.isfinite(current_lookups)
            and math.isfinite(baseline_lookups)
            and baseline_lookups > 0
            and current_lookups > 0
            else float("nan")
        )

        current_model_evaluations = float(
            row.get("mean_read_mnist_model_evaluations", float("nan"))
        )
        baseline_model_evaluations = float(
            baseline.get("mean_read_mnist_model_evaluations", float("nan"))
        )
        model_evaluation_ratio = (
            float(current_model_evaluations / baseline_model_evaluations)
            if math.isfinite(current_model_evaluations)
            and math.isfinite(baseline_model_evaluations)
            and baseline_model_evaluations > 0
            else float("nan")
        )
        row["read_mnist_model_evaluation_ratio_vs_exact"] = model_evaluation_ratio
        row["read_mnist_model_evaluation_reduction_vs_exact"] = (
            float(1.0 - model_evaluation_ratio)
            if math.isfinite(model_evaluation_ratio)
            else float("nan")
        )

        current_true_lookups = float(
            row.get("mean_true_candidate_read_mnist_lookup_calls", float("nan"))
        )
        baseline_true_lookups = float(
            baseline.get("mean_true_candidate_read_mnist_lookup_calls", float("nan"))
        )
        true_lookup_ratio = (
            float(current_true_lookups / baseline_true_lookups)
            if math.isfinite(current_true_lookups)
            and math.isfinite(baseline_true_lookups)
            and baseline_true_lookups > 0
            else float("nan")
        )
        row["true_candidate_read_mnist_lookup_ratio_vs_exact"] = true_lookup_ratio
        row["true_candidate_read_mnist_lookup_reduction_vs_exact"] = (
            float(1.0 - true_lookup_ratio)
            if math.isfinite(true_lookup_ratio)
            else float("nan")
        )


def add_paired_accuracy_delta_intervals(
        summary_rows: List[Dict[str, Any]],
        detailed_rows: List[Dict[str, Any]],
        group_keys: Sequence[str],
        *,
        bootstrap_samples: int = 2000,
        seed: int = 42,
) -> None:
    keys_wo_threshold = [key for key in group_keys if key not in {"threshold_label", "cutoff"}]

    summary_by_group: Dict[Tuple[Any, ...], Dict[str, Any]] = {
        tuple(row[key] for key in group_keys): row
        for row in summary_rows
    }
    baseline_summary_by_group: Dict[Tuple[Any, ...], Dict[str, Any]] = {
        tuple(row[key] for key in keys_wo_threshold): row
        for row in summary_rows
        if str(row.get("threshold_label")) == "exact"
    }

    detailed_grouped: Dict[Tuple[Any, ...], Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in detailed_rows:
        group = tuple(row[key] for key in group_keys)
        detailed_grouped[group][int(row["experiment_id"])] = row

    rng = np.random.default_rng(seed)

    for row in summary_rows:
        row["accuracy_delta_q25_vs_exact"] = float("nan")
        row["accuracy_delta_q75_vs_exact"] = float("nan")
        row["accuracy_delta_ci_lower_vs_exact"] = float("nan")
        row["accuracy_delta_ci_upper_vs_exact"] = float("nan")

        current_group = tuple(row[key] for key in group_keys)
        baseline_group = tuple(row[key] for key in keys_wo_threshold)
        baseline_summary = baseline_summary_by_group.get(baseline_group)
        if baseline_summary is None:
            continue

        if str(row.get("threshold_label")) == "exact":
            row["accuracy_delta_q25_vs_exact"] = 0.0
            row["accuracy_delta_q75_vs_exact"] = 0.0
            row["accuracy_delta_ci_lower_vs_exact"] = 0.0
            row["accuracy_delta_ci_upper_vs_exact"] = 0.0
            continue

        baseline_key = tuple(list(baseline_group) + ["exact", baseline_summary.get("cutoff")])
        approx_rows_by_exp = detailed_grouped.get(current_group, {})
        exact_rows_by_exp = detailed_grouped.get(baseline_key, {})
        common_ids = sorted(set(approx_rows_by_exp).intersection(exact_rows_by_exp))
        if not common_ids:
            continue

        diffs = np.asarray(
            [
                float(approx_rows_by_exp[exp_id]["correct"]) - float(exact_rows_by_exp[exp_id]["correct"])
                for exp_id in common_ids
            ],
            dtype=float,
        )
        if diffs.size == 0:
            continue

        row["accuracy_delta_q25_vs_exact"] = float(np.quantile(diffs, 0.25))
        row["accuracy_delta_q75_vs_exact"] = float(np.quantile(diffs, 0.75))

        if diffs.size == 1:
            delta_value = float(diffs[0])
            row["accuracy_delta_ci_lower_vs_exact"] = delta_value
            row["accuracy_delta_ci_upper_vs_exact"] = delta_value
            continue

        bootstrap_means = np.empty(int(bootstrap_samples), dtype=float)
        n = int(diffs.size)
        for idx in range(int(bootstrap_samples)):
            sample_indices = rng.integers(0, n, size=n)
            bootstrap_means[idx] = float(np.mean(diffs[sample_indices]))

        row["accuracy_delta_ci_lower_vs_exact"] = float(np.quantile(bootstrap_means, 0.025))
        row["accuracy_delta_ci_upper_vs_exact"] = float(np.quantile(bootstrap_means, 0.975))


def prepare_detailed_rows(raw_runs: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    detailed_rows: List[Dict[str, Any]] = []
    for run in raw_runs:
        posterior_raw = [float(value) for value in run["posterior_raw"]]
        posterior = normalize_distribution(posterior_raw)
        branch_counts = [int(value) for value in run.get("branch_counts_raw", []) if value is not None]
        output_pool = int(sum(1 for value in posterior_raw if float(value) > EPS))
        candidate_sums = [int(value) for value in run.get("candidate_sums", [])]
        candidate_count = int(len(candidate_sums))
        posterior_mass = float(sum(posterior_raw))
        zero_mass = posterior_mass <= EPS
        if posterior and not zero_mass:
            predicted_sum: int | None = int(max(range(len(posterior)), key=lambda idx: posterior[idx]))
            confidence = float(posterior[predicted_sum])
        else:
            predicted_sum = None
            confidence = 0.0
        true_sum = int(run["true_sum"])

        true_candidate_sum = int(run.get("true_candidate_sum", true_sum))
        try:
            true_candidate_index = candidate_sums.index(true_candidate_sum)
        except ValueError:
            true_candidate_index = true_candidate_sum if 0 <= true_candidate_sum < len(posterior_raw) else -1

        fallback_true_probability = (
            float(posterior_raw[true_candidate_index])
            if 0 <= true_candidate_index < len(posterior_raw)
            else 0.0
        )
        fallback_true_normalized_probability = (
            float(posterior[true_candidate_index])
            if 0 <= true_candidate_index < len(posterior)
            else 0.0
        )
        raw_branch_counts = list(run.get("branch_counts_raw", []))
        fallback_true_branch_count = (
            raw_branch_counts[true_candidate_index]
            if 0 <= true_candidate_index < len(raw_branch_counts)
            else None
        )
        true_candidate_probability_raw = float(
            run.get("true_candidate_probability_raw", fallback_true_probability)
        )
        true_candidate_normalized_probability = (
            float(true_candidate_probability_raw / posterior_mass) if posterior_mass > 0 else 0.0
        )
        if "true_candidate_probability_raw" not in run:
            true_candidate_normalized_probability = fallback_true_normalized_probability

        true_candidate_branch_count_raw = run.get("true_candidate_branch_count", fallback_true_branch_count)
        true_candidate_branch_count = (
            float("nan") if true_candidate_branch_count_raw is None else int(true_candidate_branch_count_raw)
        )
        total_branch_count = int(sum(branch_counts))
        true_candidate_runtime_sec = float(run.get("true_candidate_runtime_sec", float("nan")))
        read_mnist_lookup_calls = read_mnist_count_from_run(
            run,
            true_candidate=False,
            model_evaluations=False,
        )
        read_mnist_model_evaluations = read_mnist_count_from_run(
            run,
            true_candidate=False,
            model_evaluations=True,
        )
        true_candidate_read_mnist_lookup_calls = read_mnist_count_from_run(
            run,
            true_candidate=True,
            model_evaluations=False,
        )
        true_candidate_read_mnist_model_evaluations = read_mnist_count_from_run(
            run,
            true_candidate=True,
            model_evaluations=True,
        )

        detailed_rows.append(
            {
                "model_id": run["model_id"],
                "visualization_group": str(run.get("visualization_group", "main")),
                "cutoff_mode": str(run.get("cutoff_mode", "global")),
                "target_accuracy": float(run.get("target_accuracy", 0.0)),
                "selected_epoch": int(run.get("selected_epoch", -1)),
                "selected_test_accuracy": float(run.get("selected_test_accuracy", 0.0)),
                "experiment_id": int(run["experiment_id"]),
                "threshold_label": run["threshold_label"],
                "cutoff": run["cutoff"],
                "compile_cutoff": run.get("compile_cutoff", run.get("cutoff")),
                "adaptive_top_k": bool(run.get("adaptive_top_k", False)),
                "posterior_mass_target": run.get("posterior_mass_target"),
                "runtime_top_k_cutoff": run.get("runtime_top_k_cutoff"),
                "mean_surviving_posterior_mass": run.get("mean_surviving_posterior_mass"),
                "adaptive_cutoff_search_runtime_sec": float(run.get("adaptive_cutoff_search_runtime_sec", 0.0)),
                "n_terms": int(run["n_terms"]),
                "true_sum": true_sum,
                "predicted_sum": predicted_sum,
                "correct": int((predicted_sum is not None) and predicted_sum == true_sum),
                "prediction_valid": int(predicted_sum is not None),
                "posterior_invalid": int(zero_mass),
                "runtime_sec": float(run["runtime_sec"]),
                "read_mnist_cache_policy": str(run.get("read_mnist_cache_policy", "unknown")),
                "read_mnist_lookup_calls": read_mnist_lookup_calls,
                "read_mnist_model_evaluations": read_mnist_model_evaluations,
                "read_mnist_precompute_runtime_sec": float(
                    run.get("read_mnist_precompute_runtime_sec", 0.0)
                ),
                "confidence": confidence,
                "posterior_mass": posterior_mass,
                "posterior_entropy": entropy_from_distribution(posterior),
                "candidate_count": candidate_count,
                "output_pool": output_pool,
                "output_pool_fraction": (float(output_pool) / candidate_count) if candidate_count > 0 else 0.0,
                "total_branch_count": total_branch_count,
                "mean_branch_count": float(mean(branch_counts)) if branch_counts else 0.0,
                "max_branch_count": int(max(branch_counts)) if branch_counts else 0,
                "zero_mass": int(zero_mass),
                "true_candidate_sum": true_candidate_sum,
                "true_candidate_probability_raw": true_candidate_probability_raw,
                "true_candidate_normalized_probability": true_candidate_normalized_probability,
                "true_candidate_branch_count": true_candidate_branch_count,
                "true_candidate_runtime_sec": true_candidate_runtime_sec,
                "true_candidate_read_mnist_lookup_calls": true_candidate_read_mnist_lookup_calls,
                "true_candidate_read_mnist_model_evaluations": (
                    true_candidate_read_mnist_model_evaluations
                ),
                "true_candidate_read_mnist_precompute_runtime_sec": float(
                    run.get("true_candidate_read_mnist_precompute_runtime_sec", 0.0)
                ),
                "true_candidate_survived": int(true_candidate_probability_raw > EPS),
                "true_candidate_branch_fraction_of_total": (
                    float(true_candidate_branch_count / total_branch_count)
                    if total_branch_count > 0 and math.isfinite(float(true_candidate_branch_count))
                    else float("nan")
                ),
                "true_candidate_runtime_fraction_of_full": (
                    float(true_candidate_runtime_sec / float(run["runtime_sec"]))
                    if float(run["runtime_sec"]) > 0 and math.isfinite(true_candidate_runtime_sec)
                    else float("nan")
                ),
                "labels": str(run["labels"]),
                "global_indices": str(run["global_indices"]),
                "image_paths": str(run["image_paths"]),
                "top_predictions": str([] if zero_mass else top_predictions(posterior, top_n)),
            }
        )
    return detailed_rows


def model_order_key(model_id: str, row: Dict[str, Any]) -> Tuple[float, float, str]:
    target = float(row.get("target_accuracy", row.get("selected_test_accuracy", 0.0)))
    achieved = float(row.get("selected_test_accuracy", row.get("target_accuracy", target)))
    return (target, achieved, str(model_id))


def ordered_model_ids(summary_rows: List[Dict[str, Any]]) -> List[str]:
    best_row_by_model: Dict[str, Dict[str, Any]] = {}
    for row in summary_rows:
        model_id = str(row["model_id"])
        current_score = float(row.get("selected_test_accuracy", row.get("target_accuracy", 0.0)))
        existing = best_row_by_model.get(model_id)
        if existing is None:
            best_row_by_model[model_id] = row
            continue

        existing_score = float(existing.get("selected_test_accuracy", existing.get("target_accuracy", 0.0)))
        if current_score > existing_score:
            best_row_by_model[model_id] = row

    return [
        model_id
        for model_id, row in sorted(best_row_by_model.items(), key=lambda item: model_order_key(item[0], item[1]))
    ]


def metric_matrix(
        rows: List[Dict[str, Any]],
        metric_key: str,
        model_id: str,
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
) -> np.ndarray:
    matrix = np.full((len(term_counts), len(threshold_order)), np.nan, dtype=float)
    row_index = {term: idx for idx, term in enumerate(term_counts)}
    col_index = {label: idx for idx, label in enumerate(threshold_order)}
    for row in rows:
        if str(row["model_id"]) != model_id:
            continue
        term = int(row["n_terms"])
        label = str(row["threshold_label"])
        if term in row_index and label in col_index:
            matrix[row_index[term], col_index[label]] = float(row[metric_key])
    return matrix


def adaptive_topk_search_rows(raw_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()
    for run in raw_runs:
        state = run.get("adaptive_cutoff_state")
        if not isinstance(state, dict):
            continue
        evaluations = state.get("evaluations")
        if not isinstance(evaluations, list) or not evaluations:
            continue
        runtime_cutoff_for_key = state.get("runtime_top_k_cutoff")
        key = (
            str(run.get("model_id")),
            str(run.get("cutoff_mode", "global")),
            int(run.get("n_terms", -1)),
            str(run.get("threshold_label")),
            tuple(state.get("probe_experiment_ids") or []),
            None if runtime_cutoff_for_key is None else float(runtime_cutoff_for_key),
        )
        if key in seen:
            continue
        seen.add(key)
        for evaluation_index, evaluation in enumerate(evaluations):
            cutoff = evaluation.get("cutoff")
            surviving_mass = evaluation.get("mean_surviving_mass")
            rows.append(
                {
                    "model_id": str(run.get("model_id")),
                    "cutoff_mode": str(run.get("cutoff_mode", "global")),
                    "n_terms": int(run.get("n_terms", -1)),
                    "threshold_label": str(run.get("threshold_label")),
                    "artifact_threshold_label": str(run.get("artifact_threshold_label", "")),
                    "posterior_mass_target": state.get("posterior_mass_target", run.get("posterior_mass_target")),
                    "runtime_top_k_cutoff": state.get("runtime_top_k_cutoff", run.get("runtime_top_k_cutoff")),
                    "selected_mean_surviving_posterior_mass": state.get("mean_surviving_posterior_mass"),
                    "selected_abs_error": state.get("abs_error"),
                    "iterations": state.get("iterations"),
                    "converged": state.get("converged"),
                    "probe_experiments": state.get("probe_experiments"),
                    "search_runtime_sec": state.get("search_runtime_sec"),
                    "evaluation_index": int(evaluation_index),
                    "candidate_cutoff": None if cutoff is None else float(cutoff),
                    "mean_surviving_mass": None if surviving_mass is None else float(surviving_mass),
                }
            )
    return rows
