from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_palette import (
    A4_PAGE_WIDTH_IN,
    ACCURACY_DIVERGING,
    FIGURE_DPI,
    MODEL_ACCURACY_COLORS,
    SPEEDUP_DIVERGING,
    TRADEOFF_COLORS,
    BLUE,
)
from pipeline1_analysis import (
    compact_model_name,
    metric_matrix,
    model_label,
    model_order_key,
    non_exact_threshold_labels,
    ordered_model_ids,
    pretty_threshold_label,
    threshold_sort_key,
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

    fig = plt.figure(figsize=(max(A4_PAGE_WIDTH_IN, 5.8 * ncols + 0.9), 4.3 * nrows), constrained_layout=True)
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
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def heatmap_specs() -> List[HeatmapSpec]:
    return [
        HeatmapSpec(
            key="accuracy",
            title="Raw MNIST sum accuracy by model, term count, and cutoff",
            colorbar_label="Accuracy",
            filename="heatmap_accuracy_by_model.png",
            cmap_name="cividis",
            higher_is_better=True,
            fixed_range=(0.0, 1.0),
            fmt=".2f",
        ),
        HeatmapSpec(
            key="mean_output_pool_fraction",
            title="Surviving output-pool fraction by model, term count, and cutoff",
            colorbar_label="Output-pool fraction",
            filename="heatmap_output_pool_by_model.png",
            cmap_name="cividis",
            higher_is_better=True,
            fixed_range=(0.0, 1.0),
            fmt=".2f",
        ),
        HeatmapSpec(
            key="mean_total_branch_count",
            title="Mean total branch count by model, term count, and cutoff",
            colorbar_label="Branch count",
            filename="heatmap_branch_count_by_model.png",
            cmap_name="cividis_r",
            higher_is_better=False,
            fmt=".0f",
        ),
        HeatmapSpec(
            key="zero_mass_rate",
            title="Posterior collapse rate by model, term count, and cutoff",
            colorbar_label="Collapse rate",
            filename="heatmap_collapse_rate_by_model.png",
            cmap_name="cividis_r",
            higher_is_better=False,
            fixed_range=(0.0, 1.0),
            fmt=".2f",
        ),
        HeatmapSpec(
            key="speedup_vs_exact",
            title="Speedup vs exact baseline by model, term count, and cutoff",
            colorbar_label="Speedup vs exact (log scale)",
            filename="heatmap_speedup_by_model.png",
            cmap_name="cividis",
            higher_is_better=True,
            use_log_norm=True,
            fmt=".2f",
        ),
        HeatmapSpec(
            key="mean_true_candidate_normalized_probability",
            title="Mean normalized probability assigned to the true sum",
            colorbar_label="P(true sum | candidates)",
            filename="heatmap_true_candidate_probability_by_model.png",
            cmap_name="cividis",
            higher_is_better=True,
            fixed_range=(0.0, 1.0),
            fmt=".2f",
        ),
        HeatmapSpec(
            key="true_candidate_survival_rate",
            title="True-sum survival rate by model, term count, and cutoff",
            colorbar_label="Survival rate",
            filename="heatmap_true_candidate_survival_by_model.png",
            cmap_name="cividis",
            higher_is_better=True,
            fixed_range=(0.0, 1.0),
            fmt=".2f",
        ),
        HeatmapSpec(
            key="mean_true_candidate_branch_count",
            title="Mean branch count for the true-sum query",
            colorbar_label="True-sum branch count",
            filename="heatmap_true_candidate_branch_count_by_model.png",
            cmap_name="cividis_r",
            higher_is_better=False,
            fmt=".0f",
        ),
        HeatmapSpec(
            key="true_candidate_speedup_vs_exact",
            title="True-sum-only speedup vs exact baseline",
            colorbar_label="True-sum speedup vs exact (log scale)",
            filename="heatmap_true_candidate_speedup_by_model.png",
            cmap_name="cividis",
            higher_is_better=True,
            use_log_norm=True,
            fmt=".2f",
        ),
    ]


def build_model_styles(summary_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    model_ids = ordered_model_ids(summary_rows)
    styles: Dict[str, Dict[str, Any]] = {}
    for idx, model_id in enumerate(model_ids):
        rows = [row for row in summary_rows if str(row["model_id"]) == model_id]
        styles[model_id] = {
            "color": MODEL_ACCURACY_COLORS[idx % len(MODEL_ACCURACY_COLORS)],
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
        band_alpha: float = 0.14,
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
            ax.fill_between(x[finite_band], lower[finite_band], upper[finite_band], color=color, alpha=band_alpha, linewidth=0)

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
            color=TRADEOFF_COLORS["mass_marker"],
            markerfacecolor=TRADEOFF_COLORS["mass_marker"],
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
        color=TRADEOFF_COLORS["baseline"],
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
        *,
        title: str = "Runtime–accuracy tradeoff relative to exact inference",
) -> None:
    if not summary_rows:
        return

    approx_thresholds = positive_cutoff_thresholds(summary_rows, threshold_order)
    model_ids = ordered_model_ids(summary_rows)
    if not approx_thresholds or not model_ids or not term_counts:
        return

    tick_values = fixed_cutoff_tick_values(approx_thresholds)
    include_zero = any(abs(value) < 1e-12 for value in tick_values)
    all_x_values = [
        value
        for row in summary_rows
        if str(row.get("threshold_label")) in approx_thresholds
        and (value := cutoff_x_value_for_row(row)) is not None
    ]
    if not all_x_values:
        return

    speedup_color = TRADEOFF_COLORS["speedup"]
    accuracy_color = TRADEOFF_COLORS["accuracy"]
    combined_color = TRADEOFF_COLORS["score"]
    baseline_color = TRADEOFF_COLORS["baseline"]
    positive_zone_color = TRADEOFF_COLORS["positive_zone"]

    nrows = len(model_ids)
    ncols = len(term_counts)
    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(A4_PAGE_WIDTH_IN, 5.2 * ncols), 3.55 * nrows),
        squeeze=False,
        constrained_layout=False,
        sharex=False,
        sharey=True,
    )

    right_axes: List[Any] = []
    model_styles = build_model_styles(summary_rows)
    row_labels = [model_styles[mid]["label"] for mid in model_ids]

    for row_idx, model_id in enumerate(model_ids):
        for col_idx, n_terms in enumerate(term_counts):
            ax = axes_grid[row_idx, col_idx]
            ax_right = ax.twinx()
            right_axes.append(ax_right)

            ax.axhspan(1.0, 1.5, color=positive_zone_color, alpha=0.035, zorder=0)
            ax.axhline(1.0, color=baseline_color, linestyle="--", linewidth=1.0, alpha=0.85, zorder=1)

            rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
            row_by_label = {str(row["threshold_label"]): row for row in rows}
            exact_row = row_by_label.get("exact")

            speedup_points: List[Dict[str, Any]] = []
            accuracy_points: List[Dict[str, Any]] = []
            combined_points: List[Dict[str, Any]] = []

            exact_accuracy = finite_float_or_none(exact_row.get("accuracy")) if exact_row is not None else None
            if exact_accuracy is not None and exact_accuracy > 0.0:
                for order_idx, label in enumerate(approx_thresholds):
                    row = row_by_label.get(label)
                    if row is None:
                        continue
                    x_value = cutoff_x_value_for_row(row)
                    speedup_value = speedup_vs_exact_from_row(row)
                    accuracy_value = finite_float_or_none(row.get("accuracy"))
                    if x_value is None or speedup_value is None or accuracy_value is None:
                        continue
                    accuracy_retained_pct = 100.0 * float(accuracy_value / exact_accuracy)
                    common = {
                        "label": label,
                        "order_idx": int(order_idx),
                        "x": float(x_value),
                        "adaptive": is_adaptive_threshold_row(row),
                    }
                    speedup_points.append({**common, "y": float(speedup_value)})
                    accuracy_points.append({**common, "y": float(accuracy_retained_pct)})
                    combined_points.append(
                        {**common, "y": float(0.5 * (float(speedup_value) + accuracy_retained_pct / 100.0))}
                    )

            speedup_points.sort(key=lambda item: (float(item["x"]), int(item["order_idx"])))
            accuracy_points.sort(key=lambda item: (float(item["x"]), int(item["order_idx"])))
            combined_points.sort(key=lambda item: (float(item["x"]), int(item["order_idx"])))

            plot_cutoff_metric_series(
                ax,
                speedup_points,
                color=speedup_color,
                linewidth=2.0,
                alpha=0.96,
            )
            plot_cutoff_metric_series(
                ax_right,
                accuracy_points,
                color=accuracy_color,
                linewidth=2.0,
                alpha=0.96,
            )
            combined_line_start = len(ax.lines)
            plot_cutoff_metric_series(
                ax,
                combined_points,
                color=combined_color,
                linewidth=1.7,
                alpha=0.95,
            )
            # Restyle only the newly added combined-score line; keep its markers unchanged.
            if len(ax.lines) > combined_line_start:
                ax.lines[-1].set_linestyle("--")

            configure_numeric_cutoff_axis(
                ax,
                tick_values=tick_values,
                point_values=all_x_values,
                include_zero=include_zero,
            )
            ax.set_ylim(0.0, 1.5)
            ax_right.set_ylim(0.0, 150.0)
            ax.grid(axis="y", alpha=0.32)
            ax.grid(axis="x", alpha=0.12)

            ax.spines["left"].set_color(speedup_color)
            ax.spines["left"].set_linewidth(1.0)
            ax.tick_params(axis="y", colors=speedup_color)

            ax_right.spines["right"].set_color(accuracy_color)
            ax_right.spines["right"].set_linewidth(1.0)
            ax_right.tick_params(axis="y", colors=accuracy_color)

            if row_idx == 0:
                ax.set_title(f"{int(n_terms)} terms", pad=10)
            if row_idx < nrows - 1:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel("Cutoff")

            if col_idx != 0:
                ax.tick_params(axis="y", labelleft=False)
            else:
                ax.set_yticks([0.0, 0.5, 1.0, 1.5])
                ax.set_yticklabels(["0", "0.5", "1.0", "1.5"], color=speedup_color)

            if col_idx != ncols - 1:
                ax_right.tick_params(axis="y", labelright=False, right=False)
                ax_right.spines["right"].set_visible(False)
            else:
                ax_right.set_yticks([0.0, 50.0, 100.0, 150.0])
                ax_right.set_yticklabels(["0%", "50%", "100%", "150%"], color=accuracy_color)

    legend_handles = [
        Line2D([0], [0], color=speedup_color, marker="o", markerfacecolor=speedup_color, markeredgecolor="white", linewidth=2.0, markersize=6, label="Speedup"),
        Line2D([0], [0], color=accuracy_color, marker="o", markerfacecolor=accuracy_color, markeredgecolor="white", linewidth=2.0, markersize=6, label="Accuracy"),
        Line2D([0], [0], color=combined_color, linestyle="--", linewidth=1.8, label="Mean score"),
        Line2D([0], [0], color=baseline_color, linestyle="--", linewidth=1.0, label="Exact baseline"),
        Line2D([0], [0], marker="*", color=TRADEOFF_COLORS["mass_marker"], markerfacecolor=TRADEOFF_COLORS["mass_marker"], markeredgecolor="white", linewidth=0, markersize=12, label="mass 0.8"),
    ]

    fig.subplots_adjust(left=0.10, right=0.90, top=0.83, bottom=0.10, wspace=0.16, hspace=0.18)
    fig.suptitle(title, fontsize=16, y=0.955)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=5,
        fontsize=8.5,
        frameon=True,
        borderpad=0.6,
        handlelength=2.2,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    fig.text(0.028, 0.515, "Speedup", rotation=90, ha="center", va="center", fontsize=12, color=speedup_color)
    fig.text(0.972, 0.515, "Accuracy retained", rotation=-90, ha="center", va="center", fontsize=12, color=accuracy_color)

    for row_idx, label in enumerate(row_labels):
        bbox = axes_grid[row_idx, 0].get_position(fig)
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(0.060, y_center, label, ha="right", va="center", fontsize=11, color="#333333")

    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.10)
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
        figsize=(max(A4_PAGE_WIDTH_IN, 5.9 * ncols), 4.5 * nrows),
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
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.12)
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
        figsize=(max(A4_PAGE_WIDTH_IN, 6.3 * ncols), 4.65 * nrows),
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
                band_alpha=0.10,
                linewidth=1.95,
            )
        ax.set_title(f"{int(n_terms)} terms", pad=8)
        configure_numeric_cutoff_axis(
            ax,
            tick_values=tick_values,
            point_values=all_x_values,
            include_zero=include_zero,
        )
        ax.set_xlabel("Cutoff")
        ax.set_ylabel("Median runtime (s)")
        ax.set_yscale("log")
        ax.grid(alpha=0.35)

    finish_panel_grid(fig, axes, len(term_counts))
    handles = model_line_handles(model_ids, model_styles)
    handles.append(exact_reference_handle("Exact runtime"))
    handles.extend(adaptive_marker_handles(approx_thresholds))

    fig.subplots_adjust(top=0.78, left=0.07, right=0.985, bottom=0.10, hspace=0.32, wspace=0.24)
    fig.suptitle("Median runtime vs pruning threshold", fontsize=16, y=0.96)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=min(5, len(handles)),
        fontsize=8.7,
        frameon=True,
        borderpad=0.6,
        handlelength=2.0,
        handletextpad=0.55,
        columnspacing=1.15,
    )
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.12)
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

    all_bounds: List[float] = []
    for row in summary_rows:
        label = str(row.get("threshold_label"))
        if label not in positive_thresholds:
            continue
        for key in ("accuracy_delta_ci_lower_vs_exact", "accuracy_delta_ci_upper_vs_exact", "accuracy_delta_vs_exact"):
            value = finite_float_or_none(row.get(key))
            if value is not None:
                all_bounds.append(100.0 * float(value))
    max_abs_delta = max(5.0, max((abs(value) for value in all_bounds), default=5.0))
    ylim = (-1.15 * max_abs_delta, 1.15 * max_abs_delta)

    nrows, ncols = term_panel_grid(term_counts)
    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(A4_PAGE_WIDTH_IN, 6.3 * ncols), 4.75 * nrows),
        squeeze=False,
        constrained_layout=False,
    )
    axes = list(axes_grid.flatten())

    for ax, n_terms in zip(axes, term_counts):
        endpoints: List[Dict[str, Any]] = []
        ax.axhline(0.0, color=TRADEOFF_COLORS["baseline"], linestyle="--", linewidth=1.0, alpha=0.85)
        for model_id in model_ids:
            rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
            row_by_label = {str(row["threshold_label"]): row for row in rows}
            points = cutoff_series_points(
                row_by_label,
                positive_thresholds,
                "accuracy_delta_vs_exact",
                lower_key="accuracy_delta_ci_lower_vs_exact",
                upper_key="accuracy_delta_ci_upper_vs_exact",
                value_scale=100.0,
            )
            color = model_styles[model_id]["color"]
            plot_cutoff_metric_series(
                ax,
                points,
                color=color,
                linewidth=1.9,
                show_band=True,
                band_alpha=0.12,
            )
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
        ax.set_title(f"{int(n_terms)} terms", pad=8)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Cutoff")
        ax.set_ylabel("Δ accuracy vs exact (pp)")
        ax.grid(alpha=0.35)

    finish_panel_grid(fig, axes, len(term_counts))
    handles = model_line_handles(model_ids, model_styles)
    handles.extend([
        exact_reference_handle("Exact baseline"),
        Line2D([0], [0], marker="*", color=TRADEOFF_COLORS["mass_marker"], markerfacecolor=TRADEOFF_COLORS["mass_marker"], markeredgecolor="white", linewidth=0, markersize=11, label="mass 0.8"),
        Line2D([0], [0], color=TRADEOFF_COLORS["ci_band"], linewidth=5.0, alpha=0.12, label="95% CI"),
    ])

    fig.subplots_adjust(top=0.78, left=0.07, right=0.955, bottom=0.10, hspace=0.32, wspace=0.24)
    fig.suptitle("Accuracy change relative to exact inference", fontsize=16, y=0.96)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=min(6, len(handles)),
        fontsize=8.7,
        frameon=True,
        borderpad=0.6,
        handlelength=2.0,
        handletextpad=0.55,
        columnspacing=1.15,
    )
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.12)
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

    fig, ax = plt.subplots(figsize=(A4_PAGE_WIDTH_IN, 4.8), constrained_layout=True)
    bars = ax.bar(x, heights, width=0.62, color=BLUE, edgecolor="white", linewidth=0.9)
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

    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.12)
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

    fig, ax = plt.subplots(figsize=(max(A4_PAGE_WIDTH_IN, 8.4), 4.9), constrained_layout=True)
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
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.12)
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
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(max(A4_PAGE_WIDTH_IN, 5.4 * ncols), 4.2 * nrows), squeeze=False)
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
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
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
        "speedup_diverging", SPEEDUP_DIVERGING, N=256
    )

    acc_vmin = min(accuracy_values) if accuracy_values else -1.0
    acc_vmax = max(accuracy_values) if accuracy_values else 1.0
    acc_bound = max(abs(acc_vmin), abs(acc_vmax), 1.0)
    acc_norm = mcolors.TwoSlopeNorm(vmin=-acc_bound, vcenter=0.0, vmax=acc_bound)
    acc_cmap = mcolors.LinearSegmentedColormap.from_list(
        "accuracy_diverging", ACCURACY_DIVERGING, N=256
    )

    fig, axes_grid = plt.subplots(
        nrows=2,
        ncols=len(term_counts),
        figsize=(max(A4_PAGE_WIDTH_IN, 4.3 * max(1, len(term_counts))), 3.4 + 0.62 * len(model_ids)),
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
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def write_tradeoff_alternatives_readme(path: Path) -> None:
    lines = [
        "Tradeoff alternatives generated by visualize_results.py",
        "",
        "Question this figure answers:",
        "How much speedup versus exact inference does each approximate setting produce, and how much sum-accuracy change does it cost?",
        "",
        "Files:",
        "- 01_tradeoff_matrix_by_terms.png  [supporting alternative view]",
        "",
        "Reading guide:",
        "- Rows are models, columns are inference settings, and term counts stay in separate panels.",
        "- The top row shows speedup versus exact inference (`exact runtime / approximate runtime`); values above 1 are faster than exact, values below 1 are slower.",
        "- The bottom row shows accuracy delta relative to exact inference.",
    ]
    path.write_text("\\n".join(lines), encoding="utf-8")


def write_bundle_readme(
        path: Path,
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
        *,
        include_biased_tradeoff: bool,
) -> None:
    lines = [
        "Visualization bundle generated by visualize_results.py",
        "",
        "Main figures:",
        "- runtime_accuracy_tradeoff_by_terms.png  [main thesis-facing centerpiece figure; standard models]",
    ]
    if include_biased_tradeoff:
        lines.append(
            "- runtime_accuracy_tradeoff_biased_models_by_terms.png  "
            "[same centerpiece design; biased models only]"
        )
    lines.extend([
        "- figures/tradeoff_alternatives/01_tradeoff_matrix_by_terms.png (supporting alternative to the main tradeoff figure)",
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
    ])
    path.write_text("\n".join(lines), encoding="utf-8")







if __name__ == "__main__":
    main()
