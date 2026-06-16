from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from mnist_spll_common import ensure_dir, load_config, set_seed, stage_message
from mnist_spll_pipeline_core import (
    build_pipeline_context,
    build_stage_metadata,
    get_cutoff_modes,
    get_thresholds,
    load_json,
    stage_config_snapshot,
    write_json,
)


EPS = 1e-12
MODEL_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]


mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "grid.color": "#d9d9d9",
        "grid.linestyle": "-",
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "regular",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#cccccc",
        "legend.framealpha": 0.96,
        "font.size": 10,
    }
)


@dataclass(frozen=True)
class HeatmapSpec:
    key: str
    title: str
    colorbar_label: str
    filename: str
    cmap_name: str
    higher_is_better: bool
    use_log_norm: bool = False
    fixed_range: Tuple[float, float] | None = None
    fmt: str = ".2f"


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
        runtime_top_k_cutoffs = finite_float_values(items, "runtime_top_k_cutoff")
        surviving_masses = finite_float_values(items, "mean_surviving_posterior_mass")
        adaptive_search_runtimes = finite_float_values(items, "adaptive_cutoff_search_runtime_sec")
        true_candidate_precompute_runtimes = finite_float_values(
            items,
            "true_candidate_read_mnist_precompute_runtime_sec",
        )
        result: Dict[str, Any] = {key: first[key] for key in group_keys}
        result.update(
            {
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

        detailed_rows.append(
            {
                "model_id": run["model_id"],
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



def build_norm(spec: HeatmapSpec, matrices: Sequence[np.ndarray]) -> mcolors.Normalize:
    finite_chunks = [matrix[np.isfinite(matrix)] for matrix in matrices if np.isfinite(matrix).any()]
    if not finite_chunks:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)
    finite_values = np.concatenate(finite_chunks)

    if spec.fixed_range is not None:
        vmin, vmax = spec.fixed_range
    else:
        vmin = float(np.nanmin(finite_values))
        vmax = float(np.nanmax(finite_values))
        if math.isclose(vmin, vmax):
            vmax = vmin + 1.0

    if spec.use_log_norm:
        positive = finite_values[finite_values > 0]
        if positive.size == 0:
            return mcolors.Normalize(vmin=0.0, vmax=max(1.0, vmax))
        vmin = float(np.nanmin(positive))
        vmax = float(np.nanmax(positive))
        if math.isclose(vmin, vmax):
            vmax = vmin * 10.0
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)



def text_color_for_background(rgba: Tuple[float, float, float, float]) -> str:
    r, g, b, _ = rgba
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if luminance >= 0.58 else "white"



def format_runtime_seconds(value: float) -> str:
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}"
    if value >= 0.1:
        return f"{value:.2f}"
    if value >= 0.01:
        return f"{value:.3f}"
    if value >= 0.001:
        return f"{value:.4f}"
    return f"{value:.1e}"



def format_speedup(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"



def format_cell_value(value: float, fmt: str, metric_key: str | None = None) -> str:
    if math.isnan(value):
        return "—"
    if metric_key in {"median_runtime_sec", "median_true_candidate_runtime_sec"}:
        return format_runtime_seconds(value)
    if metric_key in {
        "speedup_vs_exact",
        "runtime_ratio_vs_exact",
        "true_candidate_speedup_vs_exact",
        "true_candidate_runtime_ratio_vs_exact",
    }:
        return format_speedup(value)
    return format(value, fmt)



def annotate_heatmap(ax, data: np.ndarray, cmap, norm, fmt: str, metric_key: str | None = None) -> None:
    n_rows, n_cols = data.shape
    for i in range(n_rows):
        for j in range(n_cols):
            value = float(data[i, j])
            if math.isnan(value):
                ax.text(j, i, "—", ha="center", va="center", color="#666666", fontsize=9)
                continue
            rgba = cmap(norm(value))
            ax.text(
                j,
                i,
                format_cell_value(value, fmt, metric_key),
                ha="center",
                va="center",
                color=text_color_for_background(rgba),
                fontsize=9,
                fontweight="medium",
            )



def annotate_heatmap_strings(ax, values: np.ndarray, labels: Sequence[Sequence[str]], cmap, norm) -> None:
    n_rows, n_cols = values.shape
    for i in range(n_rows):
        for j in range(n_cols):
            value = float(values[i, j])
            label = str(labels[i][j])
            if math.isnan(value):
                ax.text(j, i, label, ha="center", va="center", color="#666666", fontsize=9)
                continue
            rgba = cmap(norm(value))
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color=text_color_for_background(rgba),
                fontsize=9,
                fontweight="medium",
            )



def annotate_heatmap_strings(ax, values: np.ndarray, labels: Sequence[Sequence[str]], cmap, norm) -> None:
    n_rows, n_cols = values.shape
    for i in range(n_rows):
        for j in range(n_cols):
            value = float(values[i, j])
            label = str(labels[i][j])
            if math.isnan(value):
                ax.text(j, i, label, ha="center", va="center", color="#666666", fontsize=9)
                continue
            rgba = cmap(norm(value))
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color=text_color_for_background(rgba),
                fontsize=9,
                fontweight="medium",
            )



def plot_heatmap_metric(
        summary_rows: List[Dict[str, Any]],
        spec: HeatmapSpec,
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
        output_path: Path,
) -> None:
    if not summary_rows:
        return

    model_ids = ordered_model_ids(summary_rows)
    label_by_model = {
        model_id: model_label([row for row in summary_rows if str(row["model_id"]) == model_id])
        for model_id in model_ids
    }
    matrices = [metric_matrix(summary_rows, spec.key, model_id, term_counts, threshold_order) for model_id in model_ids]
    cmap = plt.get_cmap(spec.cmap_name).copy()
    cmap.set_bad("#e6e6e6")
    norm = build_norm(spec, matrices)

    n_panels = len(model_ids)
    ncols = min(2, max(1, n_panels))
    nrows = int(math.ceil(n_panels / ncols))

    fig = plt.figure(figsize=(5.8 * ncols + 0.9, 4.3 * nrows), constrained_layout=True)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols + 1, width_ratios=([1] * ncols) + [0.06])
    axes = []
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            axes.append(fig.add_subplot(gs[row_idx, col_idx]))
    cax = fig.add_subplot(gs[:, -1])

    image = None
    pretty_thresholds = [pretty_threshold_label(label) for label in threshold_order]
    for ax, model_id, matrix in zip(axes, model_ids, matrices):
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        annotate_heatmap(ax, matrix, cmap, norm, spec.fmt, spec.key)
        ax.set_title(label_by_model[model_id], fontsize=13, pad=8)
        ax.set_xticks(np.arange(len(threshold_order)))
        ax.set_xticklabels(pretty_thresholds, rotation=28, ha="right")
        ax.set_yticks(np.arange(len(term_counts)))
        ax.set_yticklabels([str(value) for value in term_counts])
        ax.set_xlabel("Cutoff")
        ax.set_ylabel("Terms")
        ax.set_xticks(np.arange(-0.5, len(threshold_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(term_counts), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

    for extra_ax in axes[len(model_ids) :]:
        extra_ax.axis("off")

    if image is not None:
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label(spec.colorbar_label, fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(spec.title, fontsize=15)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)



def heatmap_specs() -> List[HeatmapSpec]:
    return [
        HeatmapSpec(
            key="accuracy",
            title="Raw MNIST sum accuracy by model, term count, and cutoff",
            colorbar_label="Accuracy",
            filename="heatmap_accuracy_by_model.png",
            cmap_name="viridis",
            higher_is_better=True,
            fixed_range=(0.0, 1.0),
            fmt=".2f",
        ),
        HeatmapSpec(
            key="mean_output_pool_fraction",
            title="Surviving output-pool fraction by model, term count, and cutoff",
            colorbar_label="Output-pool fraction",
            filename="heatmap_output_pool_by_model.png",
            cmap_name="viridis",
            higher_is_better=True,
            fixed_range=(0.0, 1.0),
            fmt=".2f",
        ),
        HeatmapSpec(
            key="mean_total_branch_count",
            title="Mean total branch count by model, term count, and cutoff",
            colorbar_label="Branch count",
            filename="heatmap_branch_count_by_model.png",
            cmap_name="viridis_r",
            higher_is_better=False,
            fmt=".0f",
        ),
        HeatmapSpec(
            key="zero_mass_rate",
            title="Posterior collapse rate by model, term count, and cutoff",
            colorbar_label="Collapse rate",
            filename="heatmap_collapse_rate_by_model.png",
            cmap_name="viridis_r",
            higher_is_better=False,
            fixed_range=(0.0, 1.0),
            fmt=".2f",
        ),
        HeatmapSpec(
            key="speedup_vs_exact",
            title="Speedup vs exact baseline by model, term count, and cutoff",
            colorbar_label="Speedup vs exact (log scale)",
            filename="heatmap_speedup_by_model.png",
            cmap_name="viridis",
            higher_is_better=True,
            use_log_norm=True,
            fmt=".2f",
        ),
        HeatmapSpec(
            key="mean_true_candidate_normalized_probability",
            title="Mean normalized probability assigned to the true sum",
            colorbar_label="P(true sum | candidates)",
            filename="heatmap_true_candidate_probability_by_model.png",
            cmap_name="viridis",
            higher_is_better=True,
            fixed_range=(0.0, 1.0),
            fmt=".2f",
        ),
        HeatmapSpec(
            key="true_candidate_survival_rate",
            title="True-sum survival rate by model, term count, and cutoff",
            colorbar_label="Survival rate",
            filename="heatmap_true_candidate_survival_by_model.png",
            cmap_name="viridis",
            higher_is_better=True,
            fixed_range=(0.0, 1.0),
            fmt=".2f",
        ),
        HeatmapSpec(
            key="mean_true_candidate_branch_count",
            title="Mean branch count for the true-sum query",
            colorbar_label="True-sum branch count",
            filename="heatmap_true_candidate_branch_count_by_model.png",
            cmap_name="viridis_r",
            higher_is_better=False,
            fmt=".0f",
        ),
        HeatmapSpec(
            key="true_candidate_speedup_vs_exact",
            title="True-sum-only speedup vs exact baseline",
            colorbar_label="True-sum speedup vs exact (log scale)",
            filename="heatmap_true_candidate_speedup_by_model.png",
            cmap_name="viridis",
            higher_is_better=True,
            use_log_norm=True,
            fmt=".2f",
        ),
    ]



def build_model_styles(summary_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    model_ids = ordered_model_ids(summary_rows)
    cmap = plt.get_cmap("tab10")
    styles: Dict[str, Dict[str, Any]] = {}
    for idx, model_id in enumerate(model_ids):
        rows = [row for row in summary_rows if str(row["model_id"]) == model_id]
        styles[model_id] = {
            "color": cmap(idx % 10),
            "marker": MODEL_MARKERS[idx % len(MODEL_MARKERS)],
            "label": model_label(rows),
        }
    return styles



def cutoff_marker_styles(threshold_labels: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    return {
        label: {
            "marker": "*" if is_adaptive_threshold_label(str(label)) else marker_cycle[idx % len(marker_cycle)],
            "label": pretty_threshold_label(label),
        }
        for idx, label in enumerate(threshold_labels)
    }

def build_threshold_styles(threshold_order: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    approx_labels = [label for label in threshold_order if label != "exact"]
    cmap = plt.get_cmap("viridis")
    positions = np.linspace(0.18, 0.92, max(1, len(approx_labels)))
    styles: Dict[str, Dict[str, Any]] = {
        "exact": {
            "color": "#4d4d4d",
            "label": "exact",
        }
    }
    for idx, label in enumerate(approx_labels):
        styles[label] = {
            "color": cmap(float(positions[idx])),
            "label": pretty_threshold_label(label),
        }
    return styles



def get_rows(summary_rows: List[Dict[str, Any]], model_id: str, n_terms: int) -> List[Dict[str, Any]]:
    return [
        row
        for row in summary_rows
        if str(row["model_id"]) == str(model_id) and int(row["n_terms"]) == int(n_terms)
    ]



def sorted_group_rows(rows: List[Dict[str, Any]], threshold_order: Sequence[str]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: threshold_sort_key(row, threshold_order))



def term_panel_grid(term_counts: Sequence[int]) -> Tuple[int, int]:
    n_panels = len(term_counts)
    if n_panels <= 2:
        ncols = max(1, n_panels)
    elif n_panels == 4:
        ncols = 2
    else:
        ncols = min(3, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    return nrows, ncols



def finish_panel_grid(fig, axes, used_axes: int) -> None:
    for ax in axes[used_axes:]:
        ax.axis("off")



def cutoff_axis_positions(labels: Sequence[str]) -> np.ndarray:
    positions: List[float] = []
    cursor = 0.0
    for label in labels:
        positions.append(cursor)
        cursor += 1.6 if label == "exact" else 1.0
    return np.asarray(positions, dtype=float)



def positive_cutoff_thresholds(summary_rows: List[Dict[str, Any]], threshold_order: Sequence[str]) -> List[str]:
    selected: List[str] = []
    for label in threshold_order:
        if label == "exact":
            continue
        cutoff_values = [row.get("cutoff") for row in summary_rows if str(row.get("threshold_label")) == label]
        if not cutoff_values:
            continue
        adaptive_values = [
            bool(row.get("adaptive_top_k", False))
            for row in summary_rows
            if str(row.get("threshold_label")) == label
        ]
        cutoff = cutoff_values[0]
        if (cutoff is None or float(cutoff) <= 0.0) and not any(adaptive_values):
            continue
        selected.append(label)
    return selected


def is_adaptive_threshold_label(label: str) -> bool:
    return str(label).startswith("approx_mass_")


def is_adaptive_threshold_row(row: Dict[str, Any]) -> bool:
    return bool(row.get("adaptive_top_k", False)) or is_adaptive_threshold_label(
        str(row.get("threshold_label", ""))
    )


def finite_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def speedup_vs_exact_from_row(row: Dict[str, Any]) -> float | None:
    direct = finite_float_or_none(row.get("speedup_vs_exact"))
    if direct is not None and direct > 0.0:
        return direct
    runtime_ratio = finite_float_or_none(row.get("runtime_ratio_vs_exact"))
    if runtime_ratio is None or runtime_ratio <= 0.0:
        return None
    return 1.0 / runtime_ratio


def cutoff_value_from_threshold_label(label: str) -> float | None:
    if not str(label).startswith("cutoff_"):
        return None
    raw = str(label).removeprefix("cutoff_").replace("p", ".")
    return finite_float_or_none(raw)


def cutoff_x_value_for_row(row: Dict[str, Any]) -> float | None:
    """
    Return the numeric x-coordinate used by cutoff line plots.

    Fixed cutoffs are plotted at their actual configured cutoff.  Adaptive
    posterior-mass targets are plotted at the runtime TOP_K_CUTOFF selected by
    the search, so a target such as ``mass 0.8`` no longer appears as a loose
    categorical point after the numeric cutoff sweep.
    """
    if is_adaptive_threshold_row(row):
        for key in (
            "median_runtime_top_k_cutoff",
            "mean_runtime_top_k_cutoff",
            "runtime_top_k_cutoff",
        ):
            value = finite_float_or_none(row.get(key))
            if value is not None:
                return max(0.0, min(1.0, value))

    value = finite_float_or_none(row.get("cutoff"))
    if value is not None:
        return max(0.0, min(1.0, value))
    return cutoff_value_from_threshold_label(str(row.get("threshold_label", "")))


def fixed_cutoff_tick_values(threshold_labels: Sequence[str]) -> List[float]:
    values: List[float] = []
    seen: set[float] = set()
    for label in threshold_labels:
        value = cutoff_value_from_threshold_label(str(label))
        if value is None:
            continue
        rounded = round(float(value), 12)
        if rounded in seen:
            continue
        seen.add(rounded)
        values.append(float(value))
    return sorted(values)


def format_cutoff_axis_value(value: float) -> str:
    if abs(value) < 1e-12:
        return "0.0"
    value = float(value)
    if value >= 0.1:
        return f"{value:.3g}"
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10 ** exponent)
    if abs(mantissa - round(mantissa)) < 1e-9:
        mantissa = float(round(mantissa))
    if abs(mantissa - 1.0) < 1e-9:
        return f"1e{exponent}"
    return f"{mantissa:g}e{exponent}"


def configure_numeric_cutoff_axis(
        ax,
        *,
        tick_values: Sequence[float],
        point_values: Sequence[float],
        include_zero: bool,
) -> None:
    finite_points = [float(value) for value in point_values if math.isfinite(float(value))]
    finite_ticks = [float(value) for value in tick_values if math.isfinite(float(value))]
    all_values = finite_points + finite_ticks
    if not all_values:
        return

    positive_values = [value for value in all_values if value > 0.0]
    has_zero = include_zero or any(abs(value) < 1e-12 for value in all_values)
    max_value = max(all_values)

    if has_zero:
        min_positive = min(positive_values) if positive_values else 1e-3
        linthresh = max(min_positive * 0.5, 1e-9)
        ax.set_xscale("symlog", linthresh=linthresh, linscale=0.75)
        left = -linthresh * 0.35
        right = max(max_value * 1.18, min_positive * 1.5)
    elif positive_values:
        min_positive = min(positive_values)
        ax.set_xscale("log")
        left = min_positive / 1.35
        right = max(positive_values) * 1.18
    else:
        left = min(all_values) - 0.5
        right = max(all_values) + 0.5

    if not math.isclose(left, right):
        ax.set_xlim(left, right)

    ticks = sorted({round(float(value), 12): float(value) for value in finite_ticks}.values())
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([format_cutoff_axis_value(value) for value in ticks])


def cutoff_series_points(
        row_by_label: Dict[str, Dict[str, Any]],
        threshold_labels: Sequence[str],
        metric_key: str,
        *,
        lower_key: str | None = None,
        upper_key: str | None = None,
        value_scale: float = 1.0,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for order_idx, label in enumerate(threshold_labels):
        row = row_by_label.get(label)
        if row is None:
            continue
        x_value = cutoff_x_value_for_row(row)
        y_value = finite_float_or_none(row.get(metric_key))
        if x_value is None or y_value is None:
            continue
        point: Dict[str, Any] = {
            "label": label,
            "order_idx": int(order_idx),
            "x": float(x_value),
            "y": float(value_scale * y_value),
            "adaptive": is_adaptive_threshold_row(row),
        }
        if lower_key is not None:
            lower_value = finite_float_or_none(row.get(lower_key))
            point["lower"] = None if lower_value is None else float(value_scale * lower_value)
        if upper_key is not None:
            upper_value = finite_float_or_none(row.get(upper_key))
            point["upper"] = None if upper_value is None else float(value_scale * upper_value)
        points.append(point)

    points.sort(key=lambda item: (float(item["x"]), int(item["order_idx"])))
    return points


def plot_cutoff_metric_series(
        ax,
        points: Sequence[Dict[str, Any]],
        *,
        color: Any,
        label: str | None = None,
        linewidth: float = 1.8,
        alpha: float = 0.98,
        show_band: bool = False,
) -> None:
    if not points:
        return

    x = np.asarray([float(point["x"]) for point in points], dtype=float)
    y = np.asarray([float(point["y"]) for point in points], dtype=float)
    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, label=label)

    if show_band:
        lower = np.asarray([
            float(point.get("lower")) if point.get("lower") is not None else np.nan
            for point in points
        ], dtype=float)
        upper = np.asarray([
            float(point.get("upper")) if point.get("upper") is not None else np.nan
            for point in points
        ], dtype=float)
        finite_band = np.isfinite(lower) & np.isfinite(upper)
        if finite_band.any():
            ax.fill_between(x[finite_band], lower[finite_band], upper[finite_band], color=color, alpha=0.14, linewidth=0)

    fixed_points = [point for point in points if not bool(point.get("adaptive", False))]
    adaptive_points = [point for point in points if bool(point.get("adaptive", False))]
    if fixed_points:
        ax.scatter(
            [float(point["x"]) for point in fixed_points],
            [float(point["y"]) for point in fixed_points],
            marker="o",
            s=24,
            color=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
    if adaptive_points:
        ax.scatter(
            [float(point["x"]) for point in adaptive_points],
            [float(point["y"]) for point in adaptive_points],
            marker="*",
            s=132,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )


def adaptive_marker_handles(threshold_labels: Sequence[str]) -> List[Line2D]:
    labels = [label for label in threshold_labels if is_adaptive_threshold_label(str(label))]
    return [
        Line2D(
            [0],
            [0],
            marker="*",
            color="#555555",
            markerfacecolor="#555555",
            markeredgecolor="white",
            linewidth=0,
            markersize=12,
            label=f"{pretty_threshold_label(label)} (adaptive)",
        )
        for label in labels
    ]



def exact_reference_handle(label: str = "exact runtime") -> Line2D:
    return Line2D(
        [0],
        [0],
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        alpha=0.78,
        label=label,
    )


def model_line_handles(model_ids: Sequence[str], model_styles: Dict[str, Dict[str, Any]]) -> List[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=model_styles[model_id]["color"],
            marker="o",
            markerfacecolor=model_styles[model_id]["color"],
            markeredgecolor="white",
            linewidth=1.8,
            markersize=6,
            label=model_styles[model_id]["label"],
        )
        for model_id in model_ids
    ]


def cutoff_marker_handles(threshold_labels: Sequence[str], marker_styles: Dict[str, Dict[str, Any]]) -> List[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=marker_styles[label]["marker"],
            color="#555555",
            markerfacecolor="#555555",
            markeredgecolor="white",
            linewidth=0,
            markersize=9 if marker_styles[label]["marker"] != "*" else 12,
            label=marker_styles[label]["label"],
        )
        for label in threshold_labels
        if label in marker_styles
    ]


def place_horizontal_legends(
        fig,
        *,
        title: str,
        legend_rows: Sequence[Tuple[Sequence[Line2D], str | None]],
        top: float = 0.72,
        title_y: float = 0.985,
        first_legend_y: float = 0.905,
        legend_step: float = 0.080,
        max_columns: int = 7,
) -> None:
    fig.subplots_adjust(top=top, left=0.07, right=0.985, bottom=0.08, hspace=0.34, wspace=0.24)
    fig.suptitle(title, fontsize=15, y=title_y)
    for idx, (handles, row_title) in enumerate(legend_rows):
        handles = list(handles)
        if not handles:
            continue
        legend = fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, first_legend_y - idx * legend_step),
            ncol=min(max_columns, len(handles)),
            title=row_title,
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            borderaxespad=0.0,
            borderpad=0.6,
            labelspacing=0.75,
            handlelength=2.0,
            handletextpad=0.7,
            columnspacing=1.6,
        )
        fig.add_artist(legend)


def place_bottom_legends(
        fig,
        *,
        title: str,
        legend_rows: Sequence[Tuple[Sequence[Line2D], str | None]],
        bottom: float = 0.40,
        title_y: float = 0.985,
        first_legend_y: float = 0.295,
        legend_step: float = 0.145,
        max_columns: int = 8,
) -> None:
    fig.subplots_adjust(top=0.88, left=0.07, right=0.985, bottom=bottom, hspace=0.34, wspace=0.24)
    fig.suptitle(title, fontsize=15, y=title_y)
    for idx, (handles, row_title) in enumerate(legend_rows):
        handles = list(handles)
        if not handles:
            continue
        legend = fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, first_legend_y - idx * legend_step),
            ncol=min(max_columns, len(handles)),
            title=row_title,
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            borderaxespad=0.0,
            borderpad=0.6,
            labelspacing=0.75,
            handlelength=2.0,
            handletextpad=0.7,
            columnspacing=1.6,
        )
        fig.add_artist(legend)


def add_exact_runtime_reference_line(
        ax,
        row_by_label: Dict[str, Dict[str, Any]],
        metric_key: str,
        *,
        color: Any,
) -> None:
    exact_row = row_by_label.get("exact")
    if exact_row is None:
        return
    value = finite_float_or_none(exact_row.get(metric_key))
    if value is None or value <= 0.0:
        return
    ax.axhline(
        value,
        color=color,
        linestyle="--",
        linewidth=1.15,
        alpha=0.52,
        zorder=1,
    )



def spread_positions_linear(values: Sequence[float], min_gap: float) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return arr
    order = np.argsort(arr)
    adjusted = arr.copy()
    last = adjusted[order[0]]
    for idx in order[1:]:
        if adjusted[idx] - last < min_gap:
            adjusted[idx] = last + min_gap
        last = adjusted[idx]
    center_shift = float(np.mean(arr) - np.mean(adjusted))
    adjusted += center_shift
    order = np.argsort(adjusted)
    last = adjusted[order[0]]
    for idx in order[1:]:
        if adjusted[idx] - last < min_gap:
            adjusted[idx] = last + min_gap
        last = adjusted[idx]
    return adjusted



def spread_positions_log(values: Sequence[float], min_gap_decades: float = 0.05) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return arr
    positive = arr[arr > 0]
    floor = float(np.min(positive)) if positive.size else 1e-6
    safe = np.where(arr > 0, arr, floor)
    adjusted_log = spread_positions_linear(np.log10(safe), min_gap=min_gap_decades)
    return np.power(10.0, adjusted_log)



def annotate_series_right(
        ax,
        endpoints: Sequence[Dict[str, Any]],
        log_y: bool = False,
        x_pad: float = 0.36,
        min_gap: float | None = None,
        min_gap_decades: float = 0.04,
) -> None:
    if not endpoints:
        return
    y_values = [float(item["y"]) for item in endpoints]
    if log_y:
        adjusted_y = spread_positions_log(y_values, min_gap_decades=min_gap_decades)
    else:
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        default_gap = max(0.8, 0.04 * max(1.0, y_max - y_min))
        adjusted_y = spread_positions_linear(y_values, min_gap=min_gap if min_gap is not None else default_gap)
    x_values = [float(item["x"]) for item in endpoints]
    x_text = max(x_values) + x_pad
    for item, y_adj in zip(endpoints, adjusted_y):
        ax.plot(
            [float(item["x"]), x_text - 0.06],
            [float(item["y"]), float(y_adj)],
            color=item["color"],
            linewidth=0.9,
            alpha=0.55,
        )
        ax.text(
            x_text,
            float(y_adj),
            str(item["label"]),
            color=item["color"],
            va="center",
            ha="left",
            fontsize=9,
        )

def annotate_series_right_rail(
        ax,
        endpoints: Sequence[Dict[str, Any]],
        ylim: Tuple[float, float],
        x_axes: float = 1.03,
        min_gap_axes: float = 0.08,
        y_margin_axes: float = 0.06,
) -> None:
    if not endpoints:
        return

    y_min, y_max = float(ylim[0]), float(ylim[1])
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0

    sorted_endpoints = sorted(endpoints, key=lambda item: float(item["y"]))
    raw_fracs = [
        (float(item["y"]) - y_min) / (y_max - y_min)
        for item in sorted_endpoints
    ]
    raw_fracs = np.asarray(raw_fracs, dtype=float)
    raw_fracs = np.clip(raw_fracs, y_margin_axes, 1.0 - y_margin_axes)
    adjusted = spread_positions_linear(raw_fracs, min_gap=min_gap_axes)

    lower_bound = y_margin_axes
    upper_bound = 1.0 - y_margin_axes
    if adjusted.size:
        if float(np.max(adjusted)) > upper_bound:
            adjusted -= float(np.max(adjusted)) - upper_bound
        if float(np.min(adjusted)) < lower_bound:
            adjusted += lower_bound - float(np.min(adjusted))
        adjusted = np.clip(adjusted, lower_bound, upper_bound)

    for item, y_frac in zip(sorted_endpoints, adjusted):
        ax.annotate(
            str(item["label"]),
            xy=(float(item["x"]), float(item["y"])),
            xycoords="data",
            xytext=(x_axes, float(y_frac)),
            textcoords="axes fraction",
            ha="left",
            va="center",
            fontsize=9,
            color=item["color"],
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "color": item["color"],
                "linewidth": 0.9,
                "alpha": 0.55,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )



def plot_pareto_tradeoff(
        summary_rows: List[Dict[str, Any]],
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
        output_path: Path,
) -> None:
    if not summary_rows:
        return

    approx_thresholds = positive_cutoff_thresholds(summary_rows, threshold_order)
    if not approx_thresholds:
        return

    model_styles = build_model_styles(summary_rows)
    model_ids = ordered_model_ids(summary_rows)
    if not model_ids:
        return

    marker_styles = cutoff_marker_styles(approx_thresholds)
    marker_styles["exact"] = {"marker": "X", "label": "exact reference"}

    nrows, ncols = term_panel_grid(term_counts)
    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.9 * ncols, 4.45 * nrows),
        squeeze=False,
        constrained_layout=False,
    )
    axes = list(axes_grid.flatten())

    all_runtime_ratios: List[float] = []
    all_delta_pp: List[float] = []
    panel_points: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}

    for n_terms in term_counts:
        for model_id in model_ids:
            rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
            row_by_label = {str(row["threshold_label"]): row for row in rows}
            exact_row = row_by_label.get("exact")
            if exact_row is None:
                continue
            exact_runtime = finite_float_or_none(exact_row.get("median_runtime_sec"))
            exact_accuracy = finite_float_or_none(exact_row.get("accuracy"))
            if exact_runtime is None or exact_runtime <= 0.0 or exact_accuracy is None:
                continue

            points: List[Dict[str, Any]] = []
            for label in approx_thresholds:
                row = row_by_label.get(label)
                if row is None:
                    continue
                runtime = finite_float_or_none(row.get("median_runtime_sec"))
                accuracy = finite_float_or_none(row.get("accuracy"))
                if runtime is None or runtime <= 0.0 or accuracy is None:
                    continue
                runtime_ratio = runtime / exact_runtime
                delta_pp = 100.0 * (accuracy - exact_accuracy)
                points.append({"label": label, "runtime_ratio": runtime_ratio, "delta_pp": delta_pp})
                all_runtime_ratios.append(runtime_ratio)
                all_delta_pp.append(delta_pp)
            panel_points[(int(n_terms), model_id)] = points

    if not all_runtime_ratios:
        return

    for ax, n_terms in zip(axes, term_counts):
        ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0, alpha=0.75, zorder=0)
        ax.axvline(1.0, color="#777777", linestyle="--", linewidth=1.0, alpha=0.75, zorder=0)
        ax.scatter(
            [1.0],
            [0.0],
            marker="X",
            s=96,
            color="#4d4d4d",
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        for model_id in model_ids:
            points = panel_points.get((int(n_terms), model_id), [])
            if not points:
                continue
            color = model_styles[model_id]["color"]
            if len(points) >= 2:
                ax.plot(
                    [float(point["runtime_ratio"]) for point in points],
                    [float(point["delta_pp"]) for point in points],
                    color=color,
                    linewidth=1.2,
                    alpha=0.34,
                    zorder=1,
                )
            for point in points:
                marker = marker_styles[str(point["label"])]["marker"]
                ax.scatter(
                    float(point["runtime_ratio"]),
                    float(point["delta_pp"]),
                    s=132 if marker == "*" else 62,
                    color=color,
                    marker=marker,
                    edgecolors="white",
                    linewidths=0.85,
                    alpha=0.9,
                    zorder=3,
                )
        ax.set_title(f"{int(n_terms)} terms")
        ax.set_xlabel("Runtime / exact runtime")
        ax.set_ylabel("Δ sum accuracy vs exact (pp)")
        ax.set_xscale("log")
        ax.grid(alpha=0.4)

    finish_panel_grid(fig, axes, len(term_counts))

    ratio_min = min(all_runtime_ratios + [1.0])
    ratio_max = max(all_runtime_ratios + [1.0])
    if ratio_min > 0.0:
        x_min = ratio_min / 1.15
        x_max = ratio_max * 1.15
        for ax in axes[:len(term_counts)]:
            ax.set_xlim(x_min, x_max)

    max_abs_delta = max(5.0, max(abs(value) for value in all_delta_pp))
    y_lim = 1.12 * max_abs_delta
    for ax in axes[:len(term_counts)]:
        ax.set_ylim(-y_lim, y_lim)

    place_bottom_legends(
        fig,
        title="Runtime–accuracy tradeoff relative to exact inference",
        legend_rows=[
            (model_line_handles(model_ids, model_styles), "Models"),
            (cutoff_marker_handles(["exact"] + list(approx_thresholds), marker_styles), "Inference setting"),
        ],
        bottom=0.40,
        first_legend_y=0.295,
        legend_step=0.145,
        max_columns=8,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

def plot_true_candidate_metric_vs_cutoff(
        summary_rows: List[Dict[str, Any]],
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
        output_path: Path,
        *,
        metric_key: str,
        ylabel: str,
        title: str,
        yscale: str | None = None,
        ylim: Tuple[float, float] | None = None,
) -> None:
    if not summary_rows:
        return

    model_styles = build_model_styles(summary_rows)
    model_ids = ordered_model_ids(summary_rows)
    approx_thresholds = non_exact_threshold_labels(threshold_order)
    if not approx_thresholds or not model_ids:
        return
    tick_values = fixed_cutoff_tick_values(approx_thresholds)
    include_zero = any(abs(value) < 1e-12 for value in tick_values)
    all_x_values = [
        value
        for row in summary_rows
        if str(row.get("threshold_label")) in approx_thresholds
        and str(row.get("model_id")) in model_ids
        and (value := cutoff_x_value_for_row(row)) is not None
    ]

    nrows, ncols = term_panel_grid(term_counts)
    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.9 * ncols, 4.5 * nrows),
        squeeze=False,
        constrained_layout=False,
    )
    axes = list(axes_grid.flatten())

    for ax, n_terms in zip(axes, term_counts):
        for model_id in model_ids:
            rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
            row_by_label = {str(row["threshold_label"]): row for row in rows}
            points = cutoff_series_points(row_by_label, approx_thresholds, metric_key)
            color = model_styles[model_id]["color"]
            if metric_key == "median_true_candidate_runtime_sec":
                add_exact_runtime_reference_line(
                    ax,
                    row_by_label,
                    metric_key,
                    color=color,
                )
            plot_cutoff_metric_series(
                ax,
                points,
                color=color,
                label=model_styles[model_id]["label"],
            )
        ax.set_title(f"{int(n_terms)} terms")
        configure_numeric_cutoff_axis(
            ax,
            tick_values=tick_values,
            point_values=all_x_values,
            include_zero=include_zero,
        )
        ax.set_xlabel("Cutoff")
        ax.set_ylabel(ylabel)
        if yscale is not None:
            ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.45)

    finish_panel_grid(fig, axes, len(term_counts))
    legend_title = "Models"
    handles = model_line_handles(model_ids, model_styles)
    if metric_key == "median_true_candidate_runtime_sec":
        handles.append(exact_reference_handle("exact true-sum runtime"))
    handles.extend(adaptive_marker_handles(approx_thresholds))
    place_horizontal_legends(
        fig,
        title=title,
        legend_rows=[(handles, legend_title)],
        top=0.72,
        first_legend_y=0.905,
        legend_step=0.080,
        max_columns=7,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)



def plot_runtime_vs_cutoff(
        summary_rows: List[Dict[str, Any]],
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
        output_path: Path,
) -> None:
    if not summary_rows:
        return

    model_styles = build_model_styles(summary_rows)
    model_ids = ordered_model_ids(summary_rows)
    approx_thresholds = non_exact_threshold_labels(threshold_order)
    if not approx_thresholds or not model_ids:
        return
    tick_values = fixed_cutoff_tick_values(approx_thresholds)
    include_zero = any(abs(value) < 1e-12 for value in tick_values)
    all_x_values = [
        value
        for row in summary_rows
        if str(row.get("threshold_label")) in approx_thresholds
        and str(row.get("model_id")) in model_ids
        and (value := cutoff_x_value_for_row(row)) is not None
    ]

    nrows, ncols = term_panel_grid(term_counts)
    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.9 * ncols, 4.5 * nrows),
        squeeze=False,
        constrained_layout=False,
    )
    axes = list(axes_grid.flatten())

    for ax, n_terms in zip(axes, term_counts):
        for model_id in model_ids:
            rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
            row_by_label = {str(row["threshold_label"]): row for row in rows}
            points = cutoff_series_points(
                row_by_label,
                approx_thresholds,
                "median_runtime_sec",
                lower_key="runtime_q25_sec",
                upper_key="runtime_q75_sec",
            )
            color = model_styles[model_id]["color"]
            add_exact_runtime_reference_line(
                ax,
                row_by_label,
                "median_runtime_sec",
                color=color,
            )
            plot_cutoff_metric_series(
                ax,
                points,
                color=color,
                label=model_styles[model_id]["label"],
                show_band=True,
            )
        ax.set_title(f"{int(n_terms)} terms")
        configure_numeric_cutoff_axis(
            ax,
            tick_values=tick_values,
            point_values=all_x_values,
            include_zero=include_zero,
        )
        ax.set_xlabel("Cutoff")
        ax.set_ylabel("Median runtime (s)")
        ax.set_yscale("log")
        ax.grid(alpha=0.45)

    finish_panel_grid(fig, axes, len(term_counts))
    legend_title = "Models"
    handles = model_line_handles(model_ids, model_styles)
    handles.append(exact_reference_handle("exact runtime"))
    handles.extend(adaptive_marker_handles(approx_thresholds))
    place_horizontal_legends(
        fig,
        title="Median runtime vs pruning threshold",
        legend_rows=[(handles, legend_title)],
        top=0.72,
        first_legend_y=0.905,
        legend_step=0.080,
        max_columns=7,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)



def plot_accuracy_delta_vs_cutoff(
        summary_rows: List[Dict[str, Any]],
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
        output_path: Path,
) -> None:
    positive_thresholds = positive_cutoff_thresholds(summary_rows, threshold_order)
    if not summary_rows or not positive_thresholds:
        return

    model_styles = build_model_styles(summary_rows)
    model_ids = ordered_model_ids(summary_rows)
    tick_values = fixed_cutoff_tick_values(positive_thresholds)
    all_x_values = [
        value
        for row in summary_rows
        if str(row.get("threshold_label")) in positive_thresholds
        and (value := cutoff_x_value_for_row(row)) is not None
    ]

    all_deltas: List[float] = []
    for row in summary_rows:
        label = str(row.get("threshold_label"))
        if label in positive_thresholds and math.isfinite(float(row.get("accuracy_delta_vs_exact", float("nan")))):
            all_deltas.append(100.0 * float(row["accuracy_delta_vs_exact"]))
    max_abs_delta = max(5.0, max((abs(value) for value in all_deltas), default=5.0))
    ylim = (-1.12 * max_abs_delta, 1.12 * max_abs_delta)

    nrows, ncols = term_panel_grid(term_counts)
    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.9 * ncols, 4.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    axes = list(axes_grid.flatten())

    for ax, n_terms in zip(axes, term_counts):
        endpoints: List[Dict[str, Any]] = []
        ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.0)
        for model_id in model_ids:
            rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
            row_by_label = {str(row["threshold_label"]): row for row in rows}
            points = cutoff_series_points(
                row_by_label,
                positive_thresholds,
                "accuracy_delta_vs_exact",
                value_scale=100.0,
            )
            color = model_styles[model_id]["color"]
            plot_cutoff_metric_series(ax, points, color=color, linewidth=1.7)
            if points:
                last_point = max(points, key=lambda point: float(point["x"]))
                endpoints.append(
                    {
                        "x": float(last_point["x"]),
                        "y": float(last_point["y"]),
                        "label": model_styles[model_id]["label"],
                        "color": color,
                    }
                )
        configure_numeric_cutoff_axis(
            ax,
            tick_values=tick_values,
            point_values=all_x_values,
            include_zero=False,
        )
        annotate_series_right_rail(ax, endpoints, ylim=ylim, x_axes=1.03, min_gap_axes=0.085, y_margin_axes=0.08)
        ax.set_title(f"{int(n_terms)} terms")
        ax.set_ylim(*ylim)
        ax.set_xlabel("Cutoff")
        ax.set_ylabel("Δ accuracy vs exact (pp)")
        ax.grid(alpha=0.45)

    finish_panel_grid(fig, axes, len(term_counts))
    if axes:
        handles = adaptive_marker_handles(positive_thresholds)
        if handles:
            axes[0].legend(handles=handles, loc="upper left", fontsize=9)
    fig.suptitle("Accuracy change relative to exact inference", fontsize=15)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)



def zero_cutoff_label(threshold_order: Sequence[str]) -> str | None:
    for label in threshold_order:
        if not label.startswith("cutoff_"):
            continue
        raw = label.removeprefix("cutoff_").replace("p", ".")
        try:
            if float(raw) == 0.0:
                return label
        except ValueError:
            continue
    return None



def build_overhead_rows(
        summary_rows: List[Dict[str, Any]],
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
) -> List[Dict[str, Any]]:
    zero_label = zero_cutoff_label(threshold_order)
    if zero_label is None:
        return []

    output: List[Dict[str, Any]] = []
    model_ids = ordered_model_ids(summary_rows)
    for n_terms in term_counts:
        for model_id in model_ids:
            rows = {str(row["threshold_label"]): row for row in get_rows(summary_rows, model_id, n_terms)}
            if "exact" not in rows or zero_label not in rows:
                continue
            exact_runtime = float(rows["exact"]["median_runtime_sec"])
            zero_runtime = float(rows[zero_label]["median_runtime_sec"])
            ratio = float(zero_runtime / exact_runtime) if exact_runtime > 0 else float("nan")
            pct = float(100.0 * (ratio - 1.0)) if math.isfinite(ratio) else float("nan")
            output.append(
                {
                    "model_id": model_id,
                    "model_label": model_label([rows["exact"]]),
                    "n_terms": int(n_terms),
                    "exact_runtime_sec": exact_runtime,
                    "zero_cutoff_runtime_sec": zero_runtime,
                    "runtime_ratio_zero_vs_exact": ratio,
                    "runtime_delta_sec": zero_runtime - exact_runtime,
                    "percent_overhead_zero_vs_exact": pct,
                }
            )
    return output



def plot_overhead_exact_vs_zero(
        summary_rows: List[Dict[str, Any]],
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
        output_path: Path,
) -> List[Dict[str, Any]]:
    zero_label = zero_cutoff_label(threshold_order)
    if not summary_rows or zero_label is None:
        return []

    overhead_rows = build_overhead_rows(summary_rows, term_counts, threshold_order)
    if not overhead_rows:
        return []

    means = []
    for n_terms in term_counts:
        values = [
            float(row["percent_overhead_zero_vs_exact"])
            for row in overhead_rows
            if int(row["n_terms"]) == int(n_terms) and math.isfinite(float(row["percent_overhead_zero_vs_exact"]))
        ]
        if not values:
            continue
        means.append((int(n_terms), float(np.mean(values))))

    if not means:
        return overhead_rows

    x = np.arange(len(means), dtype=float)
    heights = np.array([value for _, value in means], dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    bars = ax.bar(x, heights, width=0.62, color="#4c78a8", edgecolor="white", linewidth=0.9)
    ax.axhline(0.0, color="#666666", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(term) for term, _ in means])
    ax.set_xlabel("Terms")
    ax.set_ylabel("Mean overhead at cutoff 0.0 vs exact (%)")
    ax.set_title("Approximation-path overhead at cutoff 0.0")
    ax.grid(axis="y", alpha=0.4)

    y_min = min(0.0, float(np.min(heights)))
    y_max = max(0.0, float(np.max(heights)))
    span = max(1.0, y_max - y_min)
    ax.set_ylim(y_min - 0.10 * span, y_max + 0.16 * span)

    for rect, (_, value) in zip(bars, means):
        x_pos = rect.get_x() + rect.get_width() / 2.0
        if value >= 0:
            ax.text(x_pos, value + 0.02 * span, f"{value:.0f}%", ha="center", va="bottom", fontsize=9, color="#333333")
        else:
            ax.text(x_pos, value - 0.03 * span, f"{value:.1f}%", ha="center", va="top", fontsize=9, color="#333333")

    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return overhead_rows



def unique_model_rows(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    for row in summary_rows:
        model_id = str(row["model_id"])
        existing = unique.get(model_id)
        current_score = float(row.get("selected_test_accuracy", row.get("target_accuracy", 0.0)))
        if existing is None or current_score > float(existing.get("selected_test_accuracy", existing.get("target_accuracy", 0.0))):
            unique[model_id] = row
    return [unique[mid] for mid in ordered_model_ids(summary_rows) if mid in unique]



def plot_target_vs_achieved(summary_rows: List[Dict[str, Any]], output_path: Path) -> List[Dict[str, Any]]:
    model_rows = unique_model_rows(summary_rows)
    if not model_rows:
        return []

    model_rows = sorted(model_rows, key=lambda row: model_order_key(str(row["model_id"]), row))
    x = np.arange(len(model_rows), dtype=float)
    target = np.array([100.0 * float(row.get("target_accuracy", 0.0)) for row in model_rows])
    achieved = np.array([100.0 * float(row.get("selected_test_accuracy", 0.0)) for row in model_rows])
    styles = build_model_styles(summary_rows)

    fig, ax = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    target_x = x - 0.11
    achieved_x = x + 0.11
    for idx, row in enumerate(model_rows):
        color = styles[str(row["model_id"])]["color"]
        ax.vlines(x[idx], min(target[idx], achieved[idx]), max(target[idx], achieved[idx]), color="#cfcfcf", linewidth=2.2, zorder=1)
        ax.scatter(target_x[idx], target[idx], s=68, marker="o", facecolors="white", edgecolors=color, linewidths=1.7, zorder=3)
        ax.scatter(achieved_x[idx], achieved[idx], s=64, marker="s", color=color, edgecolors="white", linewidths=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([compact_model_name(row) for row in model_rows], rotation=18, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Model")
    ax.set_title("Target vs achieved model accuracy")
    ax.grid(axis="y", alpha=0.45)
    y_min = min(float(np.min(target)), float(np.min(achieved))) - 3.0
    y_max = max(float(np.max(target)), float(np.max(achieved))) + 3.0
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.6, len(model_rows) - 0.4)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="white", markerfacecolor="white", markeredgecolor="#444444", markersize=7, linewidth=0, label="Target"),
            Line2D([0], [0], marker="s", color="white", markerfacecolor="#666666", markeredgecolor="white", markersize=7, linewidth=0, label="Achieved"),
        ],
        loc="upper left",
        ncol=2,
        columnspacing=1.3,
        handletextpad=0.5,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    return [
        {
            "model_id": row["model_id"],
            "target_accuracy": float(row.get("target_accuracy", 0.0)),
            "selected_test_accuracy": float(row.get("selected_test_accuracy", 0.0)),
            "selected_epoch": int(row.get("selected_epoch", -1)),
            "accuracy_gap": float(row.get("selected_test_accuracy", 0.0)) - float(row.get("target_accuracy", 0.0)),
        }
        for row in model_rows
    ]




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


def plot_adaptive_topk_search_iterations(
    *,
    search_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    term_counts: Sequence[int],
    value_key: str,
    ylabel: str,
    output_path: Path,
    show_target: bool = False,
    ylim: Tuple[float, float] | None = None,
) -> None:
    if not search_rows:
        return
    nrows, ncols = term_panel_grid(term_counts)
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.2 * nrows), squeeze=False)
    axes = list(axes_grid.flatten())
    model_styles = build_model_styles(summary_rows)
    threshold_styles = cutoff_marker_styles(sorted({str(row["threshold_label"]) for row in search_rows}))
    legend_handles: Dict[str, Any] = {}
    plotted_any = False

    for ax_idx, n_terms in enumerate(term_counts):
        ax = axes[ax_idx]
        term_rows = [row for row in search_rows if int(row.get("n_terms", -1)) == int(n_terms)]
        for key in sorted({(str(row["model_id"]), str(row["threshold_label"])) for row in term_rows}):
            model_id, threshold_label_value = key
            group = [row for row in term_rows if str(row["model_id"]) == model_id and str(row["threshold_label"]) == threshold_label_value]
            by_eval: Dict[int, List[float]] = defaultdict(list)
            selected_values: List[float] = []
            for row in group:
                eval_idx = int(row["evaluation_index"])
                value = row.get(value_key)
                if value is None:
                    continue
                by_eval[eval_idx].append(float(value))
                if value_key == "candidate_cutoff" and row.get("runtime_top_k_cutoff") is not None:
                    selected_values.append(float(row["runtime_top_k_cutoff"]))
            if not by_eval:
                continue
            xs = sorted(by_eval)
            ys = [float(sum(by_eval[x]) / len(by_eval[x])) for x in xs]
            style = model_styles.get(model_id, {"color": "#4d4d4d", "label": model_id})
            marker = threshold_styles.get(threshold_label_value, {"marker": "o"})["marker"]
            label = f"{style.get('label', model_id)} / {pretty_threshold_label(threshold_label_value)}"
            line, = ax.plot(xs, ys, marker=marker, markersize=4, linewidth=1.6, color=style["color"], label=label)
            legend_handles.setdefault(label, line)
            if value_key == "candidate_cutoff" and selected_values:
                selected_mean = float(sum(selected_values) / len(selected_values))
                ax.scatter([max(xs) + 0.5], [selected_mean], marker="X", s=56, color=style["color"], zorder=4)
            if show_target:
                targets = [float(row["posterior_mass_target"]) for row in group if row.get("posterior_mass_target") is not None]
                if targets:
                    ax.axhline(sum(targets) / len(targets), color=style["color"], linestyle="--", linewidth=1.0, alpha=0.55)
            plotted_any = True
        ax.set_title(f"{int(n_terms)} terms", loc="left")
        ax.set_xlabel("Cutoff-search iteration" + ("; X = selected cutoff" if value_key == "candidate_cutoff" else ""))
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    finish_panel_grid(fig, axes, len(term_counts))
    if not plotted_any:
        plt.close(fig)
        return
    title = "Adaptive top-k cutoff search path" if value_key == "candidate_cutoff" else "Adaptive top-k retained-mass search path"
    fig.suptitle(title, y=0.995)
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.04),
            ncol=min(3, len(legend_handles)),
            fontsize=8,
            frameon=True,
        )
    fig.text(
        0.01,
        0.01,
        "No rolling mean: bisection evaluations are deterministic bracket probes; smoothing would hide convergence behavior.",
        ha="left",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.14 if legend_handles else 0.035, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_matrix_by_terms(
        summary_rows: List[Dict[str, Any]],
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
        output_path: Path,
) -> None:
    if not summary_rows:
        return

    approx_thresholds = positive_cutoff_thresholds(summary_rows, threshold_order)
    model_ids = ordered_model_ids(summary_rows)
    if not approx_thresholds or not model_ids:
        return

    label_by_model = {
        model_id: model_label([row for row in summary_rows if str(row["model_id"]) == model_id])
        for model_id in model_ids
    }
    pretty_thresholds = [pretty_threshold_label(label) for label in approx_thresholds]

    speedup_matrices: List[np.ndarray] = []
    speedup_labels: List[List[List[str]]] = []
    accuracy_matrices: List[np.ndarray] = []
    accuracy_labels: List[List[List[str]]] = []
    speedup_values: List[float] = []
    accuracy_values: List[float] = []

    for n_terms in term_counts:
        speedup_matrix = np.full((len(model_ids), len(approx_thresholds)), np.nan, dtype=float)
        accuracy_matrix = np.full((len(model_ids), len(approx_thresholds)), np.nan, dtype=float)
        speedup_label_matrix = [["—" for _ in approx_thresholds] for _ in model_ids]
        accuracy_label_matrix = [["—" for _ in approx_thresholds] for _ in model_ids]

        for row_idx, model_id in enumerate(model_ids):
            rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
            row_by_label = {str(row["threshold_label"]): row for row in rows}
            for col_idx, label in enumerate(approx_thresholds):
                row = row_by_label.get(label)
                if row is None:
                    continue
                speedup = speedup_vs_exact_from_row(row)
                accuracy_delta = finite_float_or_none(row.get("accuracy_delta_vs_exact"))
                if speedup is not None:
                    speedup_matrix[row_idx, col_idx] = speedup
                    speedup_values.append(speedup)
                    speedup_label_matrix[row_idx][col_idx] = f"{speedup:.2f}×"
                if accuracy_delta is not None:
                    accuracy_delta_pp = 100.0 * accuracy_delta
                    accuracy_matrix[row_idx, col_idx] = accuracy_delta_pp
                    accuracy_values.append(accuracy_delta_pp)
                    accuracy_label_matrix[row_idx][col_idx] = f"{accuracy_delta_pp:+.0f}" if abs(accuracy_delta_pp) >= 0.5 else "0"

        speedup_matrices.append(speedup_matrix)
        speedup_labels.append(speedup_label_matrix)
        accuracy_matrices.append(accuracy_matrix)
        accuracy_labels.append(accuracy_label_matrix)

    if not speedup_values and not accuracy_values:
        return

    speedup_vmin = min(speedup_values) if speedup_values else 0.8
    speedup_vmax = max(speedup_values) if speedup_values else 1.2
    speedup_vmin = min(speedup_vmin, 1.0)
    speedup_vmax = max(speedup_vmax, 1.0)
    if math.isclose(speedup_vmin, speedup_vmax):
        speedup_vmin -= 0.1
        speedup_vmax += 0.1
    speedup_norm = mcolors.TwoSlopeNorm(vmin=speedup_vmin, vcenter=1.0, vmax=speedup_vmax)
    speedup_cmap = mcolors.LinearSegmentedColormap.from_list(
        "speedup_diverging", ["#c65a5a", "#f4eed7", "#2a9d66"], N=256
    )

    acc_vmin = min(accuracy_values) if accuracy_values else -1.0
    acc_vmax = max(accuracy_values) if accuracy_values else 1.0
    acc_bound = max(abs(acc_vmin), abs(acc_vmax), 1.0)
    acc_norm = mcolors.TwoSlopeNorm(vmin=-acc_bound, vcenter=0.0, vmax=acc_bound)
    acc_cmap = mcolors.LinearSegmentedColormap.from_list(
        "accuracy_diverging", ["#c83f49", "#f4eed7", "#53a66f"], N=256
    )

    fig, axes_grid = plt.subplots(
        nrows=2,
        ncols=len(term_counts),
        figsize=(4.3 * max(1, len(term_counts)), 3.4 + 0.62 * len(model_ids)),
        squeeze=False,
        constrained_layout=False,
    )

    speedup_img = None
    accuracy_img = None
    for col_idx, n_terms in enumerate(term_counts):
        ax_speedup = axes_grid[0, col_idx]
        ax_accuracy = axes_grid[1, col_idx]

        speedup_img = ax_speedup.imshow(
            speedup_matrices[col_idx],
            cmap=speedup_cmap,
            norm=speedup_norm,
            aspect="auto",
            interpolation="nearest",
        )
        annotate_heatmap_strings(ax_speedup, speedup_matrices[col_idx], speedup_labels[col_idx], speedup_cmap, speedup_norm)
        ax_speedup.set_title(f"{int(n_terms)} terms", fontsize=13, pad=10)
        ax_speedup.set_xticks(np.arange(len(approx_thresholds)))
        ax_speedup.set_xticklabels(pretty_thresholds, rotation=23, ha="right")
        ax_speedup.set_yticks(np.arange(len(model_ids)))
        ax_speedup.set_yticklabels([label_by_model[model_id] for model_id in model_ids] if col_idx == 0 else [])
        ax_speedup.set_xticks(np.arange(-0.5, len(approx_thresholds), 1), minor=True)
        ax_speedup.set_yticks(np.arange(-0.5, len(model_ids), 1), minor=True)
        ax_speedup.grid(which="minor", color="#f9f9f2", linestyle="-", linewidth=1.1)
        ax_speedup.tick_params(which="minor", bottom=False, left=False)

        accuracy_img = ax_accuracy.imshow(
            accuracy_matrices[col_idx],
            cmap=acc_cmap,
            norm=acc_norm,
            aspect="auto",
            interpolation="nearest",
        )
        annotate_heatmap_strings(ax_accuracy, accuracy_matrices[col_idx], accuracy_labels[col_idx], acc_cmap, acc_norm)
        ax_accuracy.set_xticks(np.arange(len(approx_thresholds)))
        ax_accuracy.set_xticklabels(pretty_thresholds, rotation=23, ha="right")
        ax_accuracy.set_yticks(np.arange(len(model_ids)))
        ax_accuracy.set_yticklabels([label_by_model[model_id] for model_id in model_ids] if col_idx == 0 else [])
        ax_accuracy.set_xlabel("Inference setting")
        ax_accuracy.set_xticks(np.arange(-0.5, len(approx_thresholds), 1), minor=True)
        ax_accuracy.set_yticks(np.arange(-0.5, len(model_ids), 1), minor=True)
        ax_accuracy.grid(which="minor", color="#f9f9f2", linestyle="-", linewidth=1.1)
        ax_accuracy.tick_params(which="minor", bottom=False, left=False)

    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.18, right=0.945, hspace=0.28, wspace=0.18)
    fig.suptitle("Tradeoff matrix relative to exact inference", fontsize=15)
    fig.text(0.065, 0.665, "Speedup vs exact\n(exact runtime /\napproximate runtime)", rotation=90, ha="center", va="center", fontsize=11)
    fig.text(0.065, 0.275, "Δ sum accuracy vs exact", rotation=90, ha="center", va="center", fontsize=11)
    if speedup_img is not None:
        cbar_speedup = fig.colorbar(speedup_img, ax=axes_grid[0, :].tolist(), fraction=0.022, pad=0.012)
        cbar_speedup.ax.tick_params(labelsize=9)
    if accuracy_img is not None:
        cbar_acc = fig.colorbar(accuracy_img, ax=axes_grid[1, :].tolist(), fraction=0.022, pad=0.012)
        cbar_acc.ax.tick_params(labelsize=9)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)



def write_tradeoff_alternatives_readme(path: Path) -> None:
    lines = [
        "Tradeoff alternatives generated by visualize_results.py",
        "",
        "Question this figure answers:",
        "How much speedup versus exact inference does each approximate setting produce, and how much sum-accuracy change does it cost?",
        "",
        "Files:",
        "- 01_tradeoff_matrix_by_terms.png  [recommended thesis-facing summary]",
        "",
        "Reading guide:",
        "- Rows are models, columns are inference settings, and term counts stay in separate panels.",
        "- The top row shows speedup versus exact inference (`exact runtime / approximate runtime`); values above 1 are faster than exact, values below 1 are slower.",
        "- The bottom row shows accuracy delta relative to exact inference.",
    ]
    path.write_text("\\n".join(lines), encoding="utf-8")



def write_bundle_readme(path: Path, term_counts: Sequence[int], threshold_order: Sequence[str]) -> None:
    lines = [
        "Visualization bundle generated by visualize_results.py",
        "",
        "Main figures:",
        "- runtime_accuracy_tradeoff_by_terms.png",
        "- figures/tradeoff_alternatives/01_tradeoff_matrix_by_terms.png (recommended alternative to the tradeoff scatter)",
        "- runtime_vs_cutoff_by_terms.png",
        "- overhead_exact_vs_zero_cutoff_by_terms.png",
        "- accuracy_delta_vs_exact_by_terms.png",
        "- true_candidate_runtime_vs_cutoff_by_terms.png",
        "- true_candidate_branch_count_vs_cutoff_by_terms.png",
        "- true_candidate_survival_vs_cutoff_by_terms.png",
        "- true_candidate_probability_vs_cutoff_by_terms.png",
        "- heatmap_branch_count_by_model.png",
        "- heatmap_collapse_rate_by_model.png",
        "",
        "Appendix / supporting figures:",
        "- target_vs_achieved_model_accuracy.png",
        "- adaptive_topk_search_cutoff_iterations_by_terms.png",
        "- adaptive_topk_search_mass_iterations_by_terms.png",
        "- heatmap_accuracy_by_model.png",
        "- heatmap_output_pool_by_model.png",
        "- heatmap_speedup_by_model.png",
        "- heatmap_true_candidate_probability_by_model.png",
        "- heatmap_true_candidate_survival_by_model.png",
        "- heatmap_true_candidate_branch_count_by_model.png",
        "- heatmap_true_candidate_speedup_by_model.png",
        "",
        "Table outputs:",
        "- detailed_results.csv",
        "- summary_results.csv",
        "- summary_results.json",
        "- overhead_exact_vs_zero_summary.csv",
        "- model_accuracy_targets.csv",
        "- adaptive_topk_search_trace.csv",
        "",
        f"Terms shown: {', '.join(str(value) for value in term_counts)}",
        f"Cutoffs shown: {', '.join(pretty_threshold_label(label) for label in threshold_order)}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")



def run_visualization_stage(config: Dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))

    ctx = build_pipeline_context(config)
    threshold_order = ordered_threshold_labels(config)
    cutoff_modes = get_cutoff_modes(config)
    cutoff_mode = cutoff_modes[0]

    stage_message(1, 3, "Loading raw staged experiments and inference runs")
    if not ctx.paths.inference_runs_path.exists():
        raise FileNotFoundError(
            f"Raw inference results not found at {ctx.paths.inference_runs_path}. Run the 'infer' step first."
        )
    experiments = load_payload_experiments(ctx.paths.staged_experiments_path)
    raw_runs = load_payload_runs(ctx.paths.inference_runs_path)

    stage_message(2, 3, "Computing derived metrics and exact-baseline deltas")
    top_n = int(ctx.inference_cfg.get("top_predictions_to_store", 5))
    detailed_rows = prepare_detailed_rows(raw_runs, top_n=top_n)
    adaptive_search_rows = adaptive_topk_search_rows(raw_runs)
    summary_by_terms = summarize_groups(
        detailed_rows,
        group_keys=["cutoff_mode", "model_id", "n_terms", "threshold_label", "cutoff"],
        threshold_order=threshold_order,
    )
    add_exact_baseline_columns(summary_by_terms, ["cutoff_mode", "model_id", "n_terms", "threshold_label", "cutoff"])

    mode_rows = [row for row in summary_by_terms if str(row.get("cutoff_mode", "global")) == cutoff_mode]
    if not mode_rows:
        raise ValueError(f"No summary rows found for cutoff mode: {cutoff_mode}")

    stage_message(3, 3, "Writing tables and figure bundle")
    vis_root = ctx.paths.visualization_root
    table_dir = ensure_dir(vis_root / "tables")
    figure_root = ensure_dir(vis_root / "figures")
    main_dir = ensure_dir(figure_root / "main_text")
    appendix_dir = ensure_dir(figure_root / "appendix")
    heatmap_dir = ensure_dir(appendix_dir / "heatmaps")
    tradeoff_alt_dir = ensure_dir(figure_root / "tradeoff_alternatives")

    write_csv(table_dir / "detailed_results.csv", detailed_rows)
    write_csv(table_dir / "summary_results.csv", summary_by_terms)
    write_csv(table_dir / "adaptive_topk_search_trace.csv", adaptive_search_rows)
    write_json(table_dir / "adaptive_topk_search_trace.json", adaptive_search_rows)
    write_json(
        table_dir / "summary_results.json",
        {
            "metadata": build_stage_metadata(
                config,
                "visualize",
                extra={
                    "num_detailed_rows": len(detailed_rows),
                    "num_summary_rows": len(summary_by_terms),
                    "num_adaptive_topk_search_rows": len(adaptive_search_rows),
                    "raw_inference_source": str(ctx.paths.inference_runs_path),
                    "cutoff_mode": cutoff_mode,
                },
            ),
            "summary_by_terms": summary_by_terms,
        },
    )

    term_counts = sorted({int(exp["n_terms"]) for exp in experiments})

    plot_pareto_tradeoff(
        summary_rows=mode_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "runtime_accuracy_tradeoff_by_terms.png",
    )
    plot_tradeoff_matrix_by_terms(
        summary_rows=mode_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=tradeoff_alt_dir / "01_tradeoff_matrix_by_terms.png",
    )
    write_tradeoff_alternatives_readme(tradeoff_alt_dir / "README.txt")
    plot_runtime_vs_cutoff(
        summary_rows=mode_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "runtime_vs_cutoff_by_terms.png",
    )
    plot_accuracy_delta_vs_cutoff(
        summary_rows=mode_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "accuracy_delta_vs_exact_by_terms.png",
    )
    plot_true_candidate_metric_vs_cutoff(
        summary_rows=mode_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "true_candidate_runtime_vs_cutoff_by_terms.png",
        metric_key="median_true_candidate_runtime_sec",
        ylabel="Median true-sum-only runtime (s)",
        title="True-sum-only runtime vs pruning threshold",
        yscale="log",
    )
    plot_true_candidate_metric_vs_cutoff(
        summary_rows=mode_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "true_candidate_branch_count_vs_cutoff_by_terms.png",
        metric_key="mean_true_candidate_branch_count",
        ylabel="Mean true-sum branch count",
        title="True-sum branch count vs pruning threshold",
    )
    plot_true_candidate_metric_vs_cutoff(
        summary_rows=mode_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "true_candidate_survival_vs_cutoff_by_terms.png",
        metric_key="true_candidate_survival_rate",
        ylabel="True-sum survival rate",
        title="True-sum survival rate vs pruning threshold",
        ylim=(0.0, 1.05),
    )
    plot_true_candidate_metric_vs_cutoff(
        summary_rows=mode_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "true_candidate_probability_vs_cutoff_by_terms.png",
        metric_key="mean_true_candidate_normalized_probability",
        ylabel="Mean normalized true-sum probability",
        title="Probability assigned to the true sum vs pruning threshold",
        ylim=(0.0, 1.05),
    )

    overhead_rows = plot_overhead_exact_vs_zero(
        summary_rows=mode_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "overhead_exact_vs_zero_cutoff_by_terms.png",
    )
    write_csv(table_dir / "overhead_exact_vs_zero_summary.csv", overhead_rows)

    target_rows = plot_target_vs_achieved(
        summary_rows=mode_rows,
        output_path=appendix_dir / "target_vs_achieved_model_accuracy.png",
    )
    write_csv(table_dir / "model_accuracy_targets.csv", target_rows)
    plot_adaptive_topk_search_iterations(
        search_rows=[row for row in adaptive_search_rows if str(row.get("cutoff_mode", "global")) == cutoff_mode],
        summary_rows=mode_rows,
        term_counts=term_counts,
        value_key="candidate_cutoff",
        ylabel="Candidate TOP_K_CUTOFF",
        output_path=appendix_dir / "adaptive_topk_search_cutoff_iterations_by_terms.png",
    )
    plot_adaptive_topk_search_iterations(
        search_rows=[row for row in adaptive_search_rows if str(row.get("cutoff_mode", "global")) == cutoff_mode],
        summary_rows=mode_rows,
        term_counts=term_counts,
        value_key="mean_surviving_mass",
        ylabel="Mean surviving posterior mass",
        output_path=appendix_dir / "adaptive_topk_search_mass_iterations_by_terms.png",
        show_target=True,
        ylim=(0.0, 1.05),
    )

    for spec in heatmap_specs():
        plot_heatmap_metric(
            summary_rows=mode_rows,
            spec=spec,
            term_counts=term_counts,
            threshold_order=threshold_order,
            output_path=heatmap_dir / spec.filename,
        )

    root_readme_lines = [
        "Visualization bundle generated by visualize_results.py",
        "",
        "Cutoff mode used: global",
        f"Terms shown: {', '.join(str(value) for value in term_counts)}",
        f"Cutoffs shown: {', '.join(pretty_threshold_label(label) for label in threshold_order)}",
        "",
        "Figures live under figures/...",
        "Tables live under tables/...",
    ]

    write_bundle_readme(figure_root / "README.txt", term_counts, threshold_order)
    (vis_root / "README.txt").write_text("\n".join(root_readme_lines), encoding="utf-8")
    stage_config_snapshot(config, vis_root / "visualize_config_used.yaml")
    print(f"Saved visualization bundle to: {vis_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute tables and figure bundles from saved raw SPLL inference runs.")
    parser.add_argument("--config", required=True, help="Path to the shared YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_visualization_stage(config)


if __name__ == "__main__":
    main()
