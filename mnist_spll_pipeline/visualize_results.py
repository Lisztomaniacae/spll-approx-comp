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
    labels: List[str] = ["exact"]
    thresholds = config.get("inference", {}).get("approximation_thresholds", [])
    for value in thresholds:
        if value is None:
            continue
        label = str(value).replace(".", "p")
        labels.append(f"cutoff_{label}")
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
    achieved_pct = int(round(100.0 * float(row.get("selected_test_accuracy", row.get("target_accuracy", 0.0)))))
    biased_suffix = " biased" if "biased" in str(row.get("model_id", "")) else ""
    label = f"{target_pct}%{biased_suffix}"
    if achieved_pct != target_pct:
        label += f" ({achieved_pct}%)"
    return label



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
        result: Dict[str, Any] = {key: first[key] for key in group_keys}
        result.update(
            {
                "target_accuracy": float(first.get("target_accuracy", 0.0)),
                "selected_epoch": int(first.get("selected_epoch", -1)),
                "selected_test_accuracy": float(first.get("selected_test_accuracy", 0.0)),
                "experiments": len(items),
                "accuracy": mean(float(item["correct"]) for item in items),
                "mean_runtime_sec": mean(runtimes),
                "median_runtime_sec": median(runtimes),
                "runtime_q25_sec": quantile(runtimes, 0.25),
                "runtime_q75_sec": quantile(runtimes, 0.75),
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
        predicted_sum = int(max(range(len(posterior)), key=lambda idx: posterior[idx])) if posterior else 0
        confidence = float(posterior[predicted_sum]) if posterior else 0.0
        branch_counts = [int(value) for value in run.get("branch_counts_raw", []) if value is not None]
        output_pool = int(sum(1 for value in posterior_raw if float(value) > EPS))
        candidate_sums = [int(value) for value in run.get("candidate_sums", [])]
        candidate_count = int(len(candidate_sums))
        posterior_mass = float(sum(posterior_raw))
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
                "n_terms": int(run["n_terms"]),
                "true_sum": true_sum,
                "predicted_sum": predicted_sum,
                "correct": int(predicted_sum == true_sum),
                "runtime_sec": float(run["runtime_sec"]),
                "confidence": confidence,
                "posterior_mass": posterior_mass,
                "posterior_entropy": entropy_from_distribution(posterior),
                "candidate_count": candidate_count,
                "output_pool": output_pool,
                "output_pool_fraction": (float(output_pool) / candidate_count) if candidate_count > 0 else 0.0,
                "total_branch_count": total_branch_count,
                "mean_branch_count": float(mean(branch_counts)) if branch_counts else 0.0,
                "max_branch_count": int(max(branch_counts)) if branch_counts else 0,
                "zero_mass": int(posterior_mass <= 0.0),
                "true_candidate_sum": true_candidate_sum,
                "true_candidate_probability_raw": true_candidate_probability_raw,
                "true_candidate_normalized_probability": true_candidate_normalized_probability,
                "true_candidate_branch_count": true_candidate_branch_count,
                "true_candidate_runtime_sec": true_candidate_runtime_sec,
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
                "top_predictions": str(top_predictions(posterior, top_n)),
            }
        )
    return detailed_rows



def model_order_key(model_id: str, row: Dict[str, Any]) -> Tuple[float, int, float, str]:
    target = float(row.get("target_accuracy", row.get("selected_test_accuracy", 0.0)))
    achieved = float(row.get("selected_test_accuracy", row.get("target_accuracy", target)))
    biased_flag = 1 if is_biased_model_id(model_id) else 0
    return (target, biased_flag, achieved, str(model_id))



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



def is_biased_model_id(model_id: str) -> bool:
    return "biased" in str(model_id)


def cutoff_marker_styles(threshold_labels: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    return {
        label: {
            "marker": marker_cycle[idx % len(marker_cycle)],
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
        cutoff = cutoff_values[0]
        if cutoff is None or float(cutoff) <= 0.0:
            continue
        selected.append(label)
    return selected



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
    term_cmap = plt.get_cmap("tab10")
    term_styles = {
        int(n_terms): {
            "color": term_cmap(idx % 10),
            "label": f"{int(n_terms)} terms",
        }
        for idx, n_terms in enumerate(term_counts)
    }
    marker_styles = cutoff_marker_styles(approx_thresholds)

    nrows, ncols = term_panel_grid(model_ids)
    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.7 * ncols, 4.4 * nrows),
        squeeze=False,
        constrained_layout=False,
    )
    axes = list(axes_grid.flatten())

    for ax, model_id in zip(axes, model_ids):
        for n_terms in term_counts:
            rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
            row_by_label = {str(row["threshold_label"]): row for row in rows}
            for label in approx_thresholds:
                row = row_by_label.get(label)
                if row is None:
                    continue
                ax.scatter(
                    float(row["median_runtime_sec"]),
                    100.0 * float(row["accuracy"]),
                    s=72,
                    color=term_styles[int(n_terms)]["color"],
                    marker=marker_styles[label]["marker"],
                    edgecolors="white",
                    linewidths=0.9,
                    alpha=0.95,
                    )
        ax.set_title(model_styles[model_id]["label"])
        ax.set_xlabel("Median inference runtime (s)")
        ax.set_ylabel("Sum accuracy (%)")
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.4)

    finish_panel_grid(fig, axes, len(model_ids))

    term_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=term_styles[int(n_terms)]["color"],
            markerfacecolor=term_styles[int(n_terms)]["color"],
            markeredgecolor="white",
            linewidth=0,
            markersize=7,
            label=term_styles[int(n_terms)]["label"],
        )
        for n_terms in term_counts
    ]
    cutoff_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_styles[label]["marker"],
            color="#555555",
            markerfacecolor="#555555",
            markeredgecolor="white",
            linewidth=0,
            markersize=7,
            label=marker_styles[label]["label"],
        )
        for label in approx_thresholds
    ]

    # Reserve a clean top band for title + legends so the subplot grid starts below them.
    fig.subplots_adjust(top=0.82, left=0.06, right=0.98, bottom=0.07, hspace=0.28, wspace=0.18)
    fig.text(0.01, 0.975, "Runtime–accuracy tradeoff by model", ha="left", va="top", fontsize=15)
    fig.legend(
        handles=term_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.975),
        ncol=min(4, len(term_handles)),
        title="Terms",
        fontsize=9,
        title_fontsize=10,
    )
    fig.legend(
        handles=cutoff_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.925),
        ncol=min(5, len(cutoff_handles)),
        title="Cutoffs",
        fontsize=9,
        title_fontsize=10,
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
        biased_only: bool | None = None,
        yscale: str | None = None,
        ylim: Tuple[float, float] | None = None,
) -> None:
    if not summary_rows:
        return

    model_styles = build_model_styles(summary_rows)
    model_ids = ordered_model_ids(summary_rows)
    if biased_only is not None:
        model_ids = [mid for mid in model_ids if is_biased_model_id(mid) == biased_only]
    approx_thresholds = non_exact_threshold_labels(threshold_order)
    if not approx_thresholds or not model_ids:
        return

    x = np.arange(len(approx_thresholds), dtype=float)
    x_labels = [pretty_threshold_label(label) for label in approx_thresholds]

    nrows, ncols = term_panel_grid(term_counts)
    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.9 * ncols, 4.5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    axes = list(axes_grid.flatten())

    for ax, n_terms in zip(axes, term_counts):
        for model_id in model_ids:
            rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
            row_by_label = {str(row["threshold_label"]): row for row in rows}
            y = np.array([
                float(row_by_label[label].get(metric_key, float("nan"))) if label in row_by_label else np.nan
                for label in approx_thresholds
            ])
            color = model_styles[model_id]["color"]
            ax.plot(
                x,
                y,
                marker="o",
                markersize=4,
                color=color,
                linewidth=1.8,
                alpha=0.98,
                label=model_styles[model_id]["label"],
            )
        ax.set_title(f"{int(n_terms)} terms")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Cutoff")
        ax.set_ylabel(ylabel)
        if yscale is not None:
            ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.45)

    finish_panel_grid(fig, axes, len(term_counts))
    if axes:
        if biased_only is None:
            legend_title = "Models"
        else:
            legend_title = "Biased models" if biased_only else "Unbiased models"
        axes[0].legend(loc="upper left", title=legend_title, fontsize=9, title_fontsize=10)
    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)



def plot_runtime_vs_cutoff(
        summary_rows: List[Dict[str, Any]],
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
        output_path: Path,
        biased_only: bool,
) -> None:
    if not summary_rows:
        return

    model_styles = build_model_styles(summary_rows)
    model_ids = [mid for mid in ordered_model_ids(summary_rows) if is_biased_model_id(mid) == biased_only]
    approx_thresholds = non_exact_threshold_labels(threshold_order)
    if not approx_thresholds or not model_ids:
        return
    x = np.arange(len(approx_thresholds), dtype=float)
    x_labels = [pretty_threshold_label(label) for label in approx_thresholds]

    nrows, ncols = term_panel_grid(term_counts)
    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.9 * ncols, 4.5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    axes = list(axes_grid.flatten())

    for ax, n_terms in zip(axes, term_counts):
        for model_id in model_ids:
            rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
            row_by_label = {str(row["threshold_label"]): row for row in rows}
            y = np.array([
                float(row_by_label[label]["median_runtime_sec"]) if label in row_by_label else np.nan
                for label in approx_thresholds
            ])
            q25 = np.array([
                float(row_by_label[label]["runtime_q25_sec"]) if label in row_by_label else np.nan
                for label in approx_thresholds
            ])
            q75 = np.array([
                float(row_by_label[label]["runtime_q75_sec"]) if label in row_by_label else np.nan
                for label in approx_thresholds
            ])
            color = model_styles[model_id]["color"]
            ax.fill_between(x, q25, q75, color=color, alpha=0.14, linewidth=0)
            ax.plot(
                x,
                y,
                marker="o",
                markersize=4,
                color=color,
                linewidth=1.8,
                alpha=0.98,
                label=model_styles[model_id]["label"],
            )
        ax.set_title(f"{int(n_terms)} terms")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Cutoff")
        ax.set_ylabel("Median runtime (s)")
        ax.set_yscale("log")
        ax.grid(alpha=0.45)

    finish_panel_grid(fig, axes, len(term_counts))
    legend_title = "Biased models" if biased_only else "Unbiased models"
    if axes:
        axes[0].legend(loc="upper left", title=legend_title, fontsize=9, title_fontsize=10)
    fig.suptitle(f"Median runtime vs pruning threshold — {legend_title.lower()}", fontsize=15)
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
    x = np.arange(len(positive_thresholds), dtype=float)
    x_labels = [pretty_threshold_label(label) for label in positive_thresholds]

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
            y = np.array([
                100.0 * float(row_by_label[label]["accuracy_delta_vs_exact"]) if label in row_by_label else np.nan
                for label in positive_thresholds
            ])
            color = model_styles[model_id]["color"]
            ax.plot(x, y, marker="o", markersize=4, color=color, linewidth=1.7)
            finite = np.isfinite(y)
            if finite.any():
                last_idx = int(np.where(finite)[0][-1])
                endpoints.append(
                    {
                        "x": float(x[last_idx]),
                        "y": float(y[last_idx]),
                        "label": model_styles[model_id]["label"],
                        "color": color,
                    }
                )
        annotate_series_right_rail(ax, endpoints, ylim=ylim, x_axes=1.03, min_gap_axes=0.085, y_margin_axes=0.08)
        ax.set_title(f"{int(n_terms)} terms")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlim(float(np.min(x)) - 0.15, float(np.max(x)) + 1.15)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Cutoff")
        ax.set_ylabel("Δ accuracy vs exact (pp)")
        ax.grid(alpha=0.45)

    finish_panel_grid(fig, axes, len(term_counts))
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
    y = np.arange(len(model_rows), dtype=float)
    target = np.array([100.0 * float(row.get("target_accuracy", 0.0)) for row in model_rows])
    achieved = np.array([100.0 * float(row.get("selected_test_accuracy", 0.0)) for row in model_rows])
    styles = build_model_styles(summary_rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    target_y = y + 0.11
    achieved_y = y - 0.11
    for idx, row in enumerate(model_rows):
        color = styles[str(row["model_id"])]["color"]
        ax.hlines(y[idx], min(target[idx], achieved[idx]), max(target[idx], achieved[idx]), color="#cfcfcf", linewidth=2.2, zorder=1)
        ax.scatter(target[idx], target_y[idx], s=68, marker="o", facecolors="white", edgecolors=color, linewidths=1.7, zorder=3)
        ax.scatter(achieved[idx], achieved_y[idx], s=64, marker="s", color=color, edgecolors="white", linewidths=0.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([compact_model_name(row) for row in model_rows])
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Target vs achieved model accuracy")
    ax.grid(axis="x", alpha=0.45)
    x_min = min(float(np.min(target)), float(np.min(achieved))) - 3.0
    x_max = max(float(np.max(target)), float(np.max(achieved))) + 3.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.6, len(model_rows) - 0.4)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="white", markerfacecolor="white", markeredgecolor="#444444", markersize=7, linewidth=0, label="Target"),
            Line2D([0], [0], marker="s", color="white", markerfacecolor="#666666", markeredgecolor="white", markersize=7, linewidth=0, label="Achieved"),
        ],
        loc="lower right",
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



def write_bundle_readme(path: Path, term_counts: Sequence[int], threshold_order: Sequence[str], cutoff_mode: str) -> None:
    lines = [
        f"Visualization bundle generated by visualize_results.py for cutoff mode: {cutoff_mode}",
        "",
        "Main figures:",
        "- runtime_accuracy_tradeoff_by_terms.png",
        "- runtime_vs_cutoff_unbiased_by_terms.png",
        "- runtime_vs_cutoff_biased_by_terms.png",
        "- overhead_exact_vs_zero_cutoff_by_terms.png",
        "- accuracy_delta_vs_exact_by_terms.png",
        "- true_candidate_runtime_vs_cutoff_unbiased_by_terms.png",
        "- true_candidate_runtime_vs_cutoff_biased_by_terms.png",
        "- true_candidate_branch_count_vs_cutoff_by_terms.png",
        "- true_candidate_survival_vs_cutoff_by_terms.png",
        "- true_candidate_probability_vs_cutoff_by_terms.png",
        "- heatmap_branch_count_by_model.png",
        "- heatmap_collapse_rate_by_model.png",
        "",
        "Appendix / supporting figures:",
        "- target_vs_achieved_model_accuracy.png",
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
    summary_by_terms = summarize_groups(
        detailed_rows,
        group_keys=["cutoff_mode", "model_id", "n_terms", "threshold_label", "cutoff"],
        threshold_order=threshold_order,
    )
    add_exact_baseline_columns(summary_by_terms, ["cutoff_mode", "model_id", "n_terms", "threshold_label", "cutoff"])

    stage_message(3, 3, "Writing tables and the global-cutoff figure bundle")
    vis_root = ctx.paths.visualization_root
    table_dir = ensure_dir(vis_root / "tables")
    figure_root = ensure_dir(vis_root / "figures")
    main_dir = ensure_dir(figure_root / "main_text")
    appendix_dir = ensure_dir(figure_root / "appendix")
    heatmap_dir = ensure_dir(appendix_dir / "heatmaps")

    write_csv(table_dir / "detailed_results.csv", detailed_rows)
    write_csv(table_dir / "summary_results.csv", summary_by_terms)
    write_json(
        table_dir / "summary_results.json",
        {
            "metadata": build_stage_metadata(
                config,
                "visualize",
                extra={
                    "num_detailed_rows": len(detailed_rows),
                    "num_summary_rows": len(summary_by_terms),
                    "raw_inference_source": str(ctx.paths.inference_runs_path),
                },
            ),
            "summary_by_terms": summary_by_terms,
        },
        )

    term_counts = sorted({int(exp["n_terms"]) for exp in experiments})

    root_readme_lines = [
        "Visualization bundle generated by visualize_results.py",
        "",
        f"Cutoff mode used: {', '.join(cutoff_modes)}",
        f"Terms shown: {', '.join(str(value) for value in term_counts)}",
        f"Cutoffs shown: {', '.join(pretty_threshold_label(label) for label in threshold_order)}",
        "",
        "Figures live under figures/global/...",
        "Tables live under tables/global/...",
    ]

    for cutoff_mode in cutoff_modes:
        mode_rows = [row for row in summary_by_terms if str(row.get("cutoff_mode", "global")) == cutoff_mode]
        if not mode_rows:
            continue
        mode_table_dir = ensure_dir(table_dir / cutoff_mode)
        mode_figure_root = ensure_dir(figure_root / cutoff_mode)
        mode_main_dir = ensure_dir(mode_figure_root / "main_text")
        mode_appendix_dir = ensure_dir(mode_figure_root / "appendix")
        mode_heatmap_dir = ensure_dir(mode_appendix_dir / "heatmaps")

        plot_pareto_tradeoff(
            summary_rows=mode_rows,
            term_counts=term_counts,
            threshold_order=threshold_order,
            output_path=mode_main_dir / "runtime_accuracy_tradeoff_by_terms.png",
        )
        plot_runtime_vs_cutoff(
            summary_rows=mode_rows,
            term_counts=term_counts,
            threshold_order=threshold_order,
            output_path=mode_main_dir / "runtime_vs_cutoff_unbiased_by_terms.png",
            biased_only=False,
        )
        plot_runtime_vs_cutoff(
            summary_rows=mode_rows,
            term_counts=term_counts,
            threshold_order=threshold_order,
            output_path=mode_main_dir / "runtime_vs_cutoff_biased_by_terms.png",
            biased_only=True,
        )
        plot_accuracy_delta_vs_cutoff(
            summary_rows=mode_rows,
            term_counts=term_counts,
            threshold_order=threshold_order,
            output_path=mode_main_dir / "accuracy_delta_vs_exact_by_terms.png",
        )
        plot_true_candidate_metric_vs_cutoff(
            summary_rows=mode_rows,
            term_counts=term_counts,
            threshold_order=threshold_order,
            output_path=mode_main_dir / "true_candidate_runtime_vs_cutoff_unbiased_by_terms.png",
            metric_key="median_true_candidate_runtime_sec",
            ylabel="Median true-sum-only runtime (s)",
            title="True-sum-only runtime vs pruning threshold — unbiased models",
            biased_only=False,
            yscale="log",
        )
        plot_true_candidate_metric_vs_cutoff(
            summary_rows=mode_rows,
            term_counts=term_counts,
            threshold_order=threshold_order,
            output_path=mode_main_dir / "true_candidate_runtime_vs_cutoff_biased_by_terms.png",
            metric_key="median_true_candidate_runtime_sec",
            ylabel="Median true-sum-only runtime (s)",
            title="True-sum-only runtime vs pruning threshold — biased models",
            biased_only=True,
            yscale="log",
        )
        plot_true_candidate_metric_vs_cutoff(
            summary_rows=mode_rows,
            term_counts=term_counts,
            threshold_order=threshold_order,
            output_path=mode_main_dir / "true_candidate_branch_count_vs_cutoff_by_terms.png",
            metric_key="mean_true_candidate_branch_count",
            ylabel="Mean true-sum branch count",
            title="True-sum branch count vs pruning threshold",
        )
        plot_true_candidate_metric_vs_cutoff(
            summary_rows=mode_rows,
            term_counts=term_counts,
            threshold_order=threshold_order,
            output_path=mode_main_dir / "true_candidate_survival_vs_cutoff_by_terms.png",
            metric_key="true_candidate_survival_rate",
            ylabel="True-sum survival rate",
            title="True-sum survival rate vs pruning threshold",
            ylim=(0.0, 1.05),
        )
        plot_true_candidate_metric_vs_cutoff(
            summary_rows=mode_rows,
            term_counts=term_counts,
            threshold_order=threshold_order,
            output_path=mode_main_dir / "true_candidate_probability_vs_cutoff_by_terms.png",
            metric_key="mean_true_candidate_normalized_probability",
            ylabel="Mean normalized true-sum probability",
            title="Probability assigned to the true sum vs pruning threshold",
            ylim=(0.0, 1.05),
        )

        overhead_rows = plot_overhead_exact_vs_zero(
            summary_rows=mode_rows,
            term_counts=term_counts,
            threshold_order=threshold_order,
            output_path=mode_main_dir / "overhead_exact_vs_zero_cutoff_by_terms.png",
        )
        write_csv(mode_table_dir / "overhead_exact_vs_zero_summary.csv", overhead_rows)

        target_rows = plot_target_vs_achieved(
            summary_rows=mode_rows,
            output_path=mode_appendix_dir / "target_vs_achieved_model_accuracy.png",
        )
        write_csv(mode_table_dir / "model_accuracy_targets.csv", target_rows)

        mode_summary_json = {
            "metadata": build_stage_metadata(
                config,
                "visualize_mode",
                extra={
                    "cutoff_mode": cutoff_mode,
                    "num_summary_rows": len(mode_rows),
                    "raw_inference_source": str(ctx.paths.inference_runs_path),
                },
            ),
            "summary_by_terms": mode_rows,
        }
        write_csv(mode_table_dir / "summary_results.csv", mode_rows)
        write_json(mode_table_dir / "summary_results.json", mode_summary_json)

        for spec in heatmap_specs():
            plot_heatmap_metric(
                summary_rows=mode_rows,
                spec=spec,
                term_counts=term_counts,
                threshold_order=threshold_order,
                output_path=mode_heatmap_dir / spec.filename,
            )

        write_bundle_readme(mode_figure_root / "README.txt", term_counts, threshold_order, cutoff_mode)

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
