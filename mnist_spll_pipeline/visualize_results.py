from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from mnist_spll_common import ensure_dir, load_config, set_seed, stage_message
from mnist_spll_pipeline_core import (
    build_pipeline_context,
    build_stage_metadata,
    load_json,
    stage_config_snapshot,
    write_json,
)


EPS = 1e-12


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
    annotate_with: str | None = None


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


def write_csv(path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_payload_runs(path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        return payload["runs"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected inference run payload at {path}")


def load_payload_experiments(path) -> List[Dict[str, Any]]:
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
        return "0"
    if value >= 0.1:
        return f"{value:.2g}"
    exponent = int(round(math.log10(value)))
    if abs(value - 10**exponent) < 1e-12:
        return f"1e{exponent}"
    mantissa = value / (10**math.floor(math.log10(value)))
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


def model_label(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "model"
    row = rows[0]
    actual_pct = 100.0 * float(row.get("selected_test_accuracy", row.get("target_accuracy", 0.0)))
    return f"{row['model_id']} ({actual_pct:.0f}%)"


def summarize_groups(
    rows: List[Dict[str, Any]],
    group_keys: Sequence[str],
    threshold_order: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    summary: List[Dict[str, Any]] = []
    for _, items in grouped.items():
        first = items[0]
        result: Dict[str, Any] = {key: first[key] for key in group_keys}
        result.update(
            {
                "target_accuracy": float(first.get("target_accuracy", 0.0)),
                "selected_epoch": int(first.get("selected_epoch", -1)),
                "selected_test_accuracy": float(first.get("selected_test_accuracy", 0.0)),
                "experiments": len(items),
                "accuracy": mean(float(item["correct"]) for item in items),
                "mean_runtime_sec": mean(float(item["runtime_sec"]) for item in items),
                "median_runtime_sec": median(float(item["runtime_sec"]) for item in items),
                "mean_confidence": mean(float(item["confidence"]) for item in items),
                "mean_output_pool": mean(float(item["output_pool"]) for item in items),
                "mean_output_pool_fraction": mean(float(item["output_pool_fraction"]) for item in items),
                "mean_total_branch_count": mean(float(item["total_branch_count"]) for item in items),
                "mean_max_branch_count": mean(float(item["max_branch_count"]) for item in items),
                "mean_posterior_entropy": mean(float(item["posterior_entropy"]) for item in items),
                "mean_posterior_mass": mean(float(item["posterior_mass"]) for item in items),
                "zero_mass_rate": mean(float(item["zero_mass"]) for item in items),
                "mean_candidate_count": mean(float(item["candidate_count"]) for item in items),
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


def add_speedup_columns(summary_rows: List[Dict[str, Any]], group_keys: Sequence[str]) -> None:
    baseline_by_group: Dict[Tuple[Any, ...], float] = {}
    keys_wo_threshold = [key for key in group_keys if key not in {"threshold_label", "cutoff"}]
    for row in summary_rows:
        if row.get("threshold_label") == "exact":
            baseline_by_group[tuple(row[key] for key in keys_wo_threshold)] = float(row["median_runtime_sec"])

    for row in summary_rows:
        base_key = tuple(row[key] for key in keys_wo_threshold)
        baseline = baseline_by_group.get(base_key)
        runtime = float(row["median_runtime_sec"])
        if baseline is None or runtime <= 0:
            row["speedup_vs_exact"] = 0.0
        else:
            row["speedup_vs_exact"] = float(baseline / runtime)


def prepare_detailed_rows(raw_runs: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    detailed_rows: List[Dict[str, Any]] = []
    for run in raw_runs:
        posterior_raw = [float(value) for value in run["posterior_raw"]]
        posterior = normalize_distribution(posterior_raw)
        predicted_sum = int(max(range(len(posterior)), key=lambda idx: posterior[idx])) if posterior else 0
        confidence = float(posterior[predicted_sum]) if posterior else 0.0
        branch_counts = [int(value) for value in run.get("branch_counts_raw", []) if value is not None]
        output_pool = int(sum(1 for value in posterior_raw if float(value) > EPS))
        candidate_count = int(len(run.get("candidate_sums", [])))
        posterior_mass = float(sum(posterior_raw))
        detailed_rows.append(
            {
                "model_id": run["model_id"],
                "target_accuracy": float(run.get("target_accuracy", 0.0)),
                "selected_epoch": int(run.get("selected_epoch", -1)),
                "selected_test_accuracy": float(run.get("selected_test_accuracy", 0.0)),
                "experiment_id": int(run["experiment_id"]),
                "threshold_label": run["threshold_label"],
                "cutoff": run["cutoff"],
                "n_terms": int(run["n_terms"]),
                "true_sum": int(run["true_sum"]),
                "predicted_sum": predicted_sum,
                "correct": int(predicted_sum == int(run["true_sum"])),
                "runtime_sec": float(run["runtime_sec"]),
                "confidence": confidence,
                "posterior_mass": posterior_mass,
                "posterior_entropy": entropy_from_distribution(posterior),
                "candidate_count": candidate_count,
                "output_pool": output_pool,
                "output_pool_fraction": (float(output_pool) / candidate_count) if candidate_count > 0 else 0.0,
                "total_branch_count": int(sum(branch_counts)),
                "mean_branch_count": float(mean(branch_counts)) if branch_counts else 0.0,
                "max_branch_count": int(max(branch_counts)) if branch_counts else 0,
                "zero_mass": int(posterior_mass <= 0.0),
                "labels": str(run["labels"]),
                "global_indices": str(run["global_indices"]),
                "image_paths": str(run["image_paths"]),
                "top_predictions": str(top_predictions(posterior, top_n)),
            }
        )
    return detailed_rows


def ordered_model_ids(summary_rows: List[Dict[str, Any]]) -> List[str]:
    scores: Dict[str, float] = {}
    for row in summary_rows:
        model_id = str(row["model_id"])
        scores.setdefault(model_id, float(row.get("selected_test_accuracy", row.get("target_accuracy", 0.0))))
    return [model_id for model_id, _ in sorted(scores.items(), key=lambda item: item[1])]


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
    finite_values = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices if np.isfinite(matrix).any()])
    if finite_values.size == 0:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)

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
    if metric_key == "median_runtime_sec":
        return format_runtime_seconds(value)
    if metric_key == "speedup_vs_exact":
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
    output_path,
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

    for extra_ax in axes[len(model_ids):]:
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
            title="MNIST sum accuracy by model, term count, and cutoff",
            colorbar_label="Accuracy",
            filename="heatmap_accuracy_by_model.png",
            cmap_name="viridis",
            higher_is_better=True,
            fixed_range=(0.0, 1.0),
            fmt=".2f",
        ),
        HeatmapSpec(
            key="median_runtime_sec",
            title="Median inference runtime by model, term count, and cutoff",
            colorbar_label="Median runtime (s, log scale)",
            filename="heatmap_median_runtime_by_model.png",
            cmap_name="viridis_r",
            higher_is_better=False,
            use_log_norm=True,
            fmt=".2g",
        ),
        HeatmapSpec(
            key="mean_confidence",
            title="Prediction confidence by model, term count, and cutoff",
            colorbar_label="Confidence",
            filename="heatmap_confidence_by_model.png",
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
    ]


def write_readme(path, term_counts: Sequence[int], threshold_order: Sequence[str]) -> None:
    lines = [
        "Heatmap bundle generated by visualize_results.py",
        "",
        "Color semantics:",
        "- Bright = better",
        "- Dark = worse",
        "- Gray = missing / not available",
        "",
        "Metric directions:",
        "- Accuracy, confidence, output-pool fraction, speedup: higher is better",
        "- Runtime, branch count, collapse rate: lower is better",
        "",
        f"Terms shown: {', '.join(str(value) for value in term_counts)}",
        f"Cutoffs shown: {', '.join(pretty_threshold_label(label) for label in threshold_order)}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_visualization_stage(config: Dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    ctx = build_pipeline_context(config)
    threshold_order = ordered_threshold_labels(config)

    stage_message(1, 3, "Loading raw staged experiments and inference runs")
    if not ctx.paths.inference_runs_path.exists():
        raise FileNotFoundError(
            f"Raw inference results not found at {ctx.paths.inference_runs_path}. Run the 'infer' step first."
        )
    experiments = load_payload_experiments(ctx.paths.staged_experiments_path)
    raw_runs = load_payload_runs(ctx.paths.inference_runs_path)

    stage_message(2, 3, "Computing derived metrics from raw posterior traces")
    top_n = int(ctx.inference_cfg.get("top_predictions_to_store", 5))
    detailed_rows = prepare_detailed_rows(raw_runs, top_n=top_n)
    summary_by_terms = summarize_groups(
        detailed_rows,
        group_keys=["model_id", "n_terms", "threshold_label", "cutoff"],
        threshold_order=threshold_order,
    )
    add_speedup_columns(summary_by_terms, ["model_id", "n_terms", "threshold_label", "cutoff"])

    stage_message(3, 3, "Writing heatmap tables and plots")
    vis_root = ctx.paths.visualization_root
    heatmap_dir = ensure_dir(vis_root / "heatmaps")
    write_csv(heatmap_dir / "heatmap_summary.csv", summary_by_terms)
    write_json(
        heatmap_dir / "heatmap_summary.json",
        {
            "metadata": build_stage_metadata(
                config,
                "visualize_heatmaps",
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
    for spec in heatmap_specs():
        plot_heatmap_metric(
            summary_rows=summary_by_terms,
            spec=spec,
            term_counts=term_counts,
            threshold_order=threshold_order,
            output_path=heatmap_dir / spec.filename,
        )

    write_readme(heatmap_dir / "README.txt", term_counts, threshold_order)
    stage_config_snapshot(config, heatmap_dir / "visualize_config_used.yaml")
    print(f"Saved heatmap visualization bundle to: {heatmap_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute heatmap summaries and plots from saved raw SPLL inference runs.")
    parser.add_argument("--config", required=True, help="Path to the shared YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_visualization_stage(config)


if __name__ == "__main__":
    main()
