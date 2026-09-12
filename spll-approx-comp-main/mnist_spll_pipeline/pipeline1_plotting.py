from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from plot_palette import (
    A4_PAGE_WIDTH_IN,
    ACCURACY_DIVERGING,
    FIGURE_DPI,
    MODEL_ACCURACY_COLORS,
    SPEEDUP_DIVERGING,
    THESIS_GRID_ALPHA,
    THESIS_LEGEND_SIZE,
    THESIS_PANEL_LABEL_SIZE,
    THESIS_TEXT_WIDTH_IN,
    TRADEOFF_COLORS,
    BLUE,
    LEGEND_BOX_ALPHA,
    LEGEND_BOX_EDGE,
    LEGEND_BOX_FACE,
    style_legend_frame,
    thesis_panel_figure_size,
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
TRADEOFF_REGIME_ORDER = (50, 70, 90)
TRADEOFF_REGIME_LINESTYLES = {50: ":", 70: "--", 90: "-"}
TRADEOFF_LINE_WIDTH = 1.95
TRADEOFF_MARKER_SIZE = 4.2
TRADEOFF_ADAPTIVE_MARKER_SIZE = 128.0


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


def metric_has_usable_values(summary_rows: Sequence[Dict[str, Any]], metric_key: str) -> bool:
    """Return whether a metric contains reader-meaningful values.

    Branch-count instrumentation is optional in archived inference artifacts.
    An all-missing branch metric or an all-zero branch metric would otherwise
    produce an empty/misleading plot that looks like a real scientific result.
    """

    values = [
        value
        for row in summary_rows
        if (value := finite_float_or_none(row.get(metric_key))) is not None
    ]
    if not values:
        return False
    if "branch_count" in str(metric_key) and not any(float(value) > 0.0 for value in values):
        return False
    return True


def plot_heatmap_metric(
        summary_rows: List[Dict[str, Any]],
        spec: HeatmapSpec,
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
        output_path: Path,
) -> None:
    if not summary_rows or not metric_has_usable_values(summary_rows, spec.key):
        output_path.unlink(missing_ok=True)
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
    else:
        # Three side-by-side panels become unreadable once the PNG is scaled to
        # the thesis text block.  Two columns keep each panel close to square;
        # for the common 2/3/4-term case the fourth cell can carry the legend.
        ncols = 2
    nrows = int(math.ceil(n_panels / ncols))
    return nrows, ncols


def finish_panel_grid(fig, axes, used_axes: int) -> None:
    for ax in axes[used_axes:]:
        ax.axis("off")


def _legend_container_patch(ax, *, x: float = 0.02, y: float = 0.08, width: float = 0.94, height: float = 0.84) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        transform=ax.transAxes,
        linewidth=0.9,
        edgecolor=LEGEND_BOX_EDGE,
        facecolor=LEGEND_BOX_FACE,
        alpha=LEGEND_BOX_ALPHA,
        zorder=0,
    )
    ax.add_patch(patch)
    return patch


def _legend_panel_positions(block_count: int) -> Tuple[List[float], List[float]]:
    if block_count <= 1:
        return [0.86], [0.78]
    if block_count == 2:
        return [0.88, 0.50], [0.80, 0.42]
    if block_count == 3:
        return [0.90, 0.62, 0.33], [0.82, 0.54, 0.25]
    return [0.91, 0.68, 0.45, 0.22], [0.83, 0.60, 0.37, 0.14]


def _draw_grouped_legend_box(
        legend_ax,
        legend_blocks: Sequence[Tuple[Sequence[Line2D], str | None, int]],
        *,
        x: float = 0.07,
        y: float = 0.08,
        width: float = 0.88,
        height: float = 0.84,
        title_fontsize: float = THESIS_LEGEND_SIZE,
        entry_fontsize: float = THESIS_LEGEND_SIZE - 0.3,
) -> None:
    nonempty_blocks: List[Tuple[List[Line2D], str | None, int]] = [
        (list(handles), title, max(1, int(ncol)))
        for handles, title, ncol in legend_blocks
        if list(handles)
    ]
    if not nonempty_blocks:
        legend_ax.axis("off")
        return

    legend_ax.axis("off")
    _legend_container_patch(legend_ax, x=x - 0.03, y=y - 0.02, width=width + 0.06, height=height + 0.04)
    title_positions, legend_positions = _legend_panel_positions(len(nonempty_blocks))
    for idx, ((handles, title, ncol), title_y, legend_y) in enumerate(zip(nonempty_blocks, title_positions, legend_positions)):
        if title:
            legend_ax.text(
                x,
                title_y,
                title,
                transform=legend_ax.transAxes,
                ha="left",
                va="top",
                fontsize=title_fontsize,
            )
        legend = legend_ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(x, legend_y),
            ncol=ncol,
            fontsize=entry_fontsize,
            frameon=False,
            borderaxespad=0.0,
            labelspacing=0.48,
            handlelength=1.9,
            handletextpad=0.55,
            columnspacing=0.95,
        )
        legend._legend_box.align = "left"
        if idx < len(nonempty_blocks) - 1:
            legend_ax.add_artist(legend)


def _draw_horizontal_grouped_legend_box(
        fig,
        *,
        bounds: Tuple[float, float, float, float],
        legend_rows: Sequence[Tuple[Sequence[Line2D], str | None, int]],
        title_x: float = 0.04,
        legend_x: float = 0.28,
        row_title_y: Sequence[float] | None = None,
        row_legend_y: Sequence[float] | None = None,
        title_fontsize: float = THESIS_LEGEND_SIZE,
        entry_fontsize: float = THESIS_LEGEND_SIZE - 0.15,
) -> None:
    nonempty_rows: List[Tuple[List[Line2D], str | None, int]] = [
        (list(handles), title, max(1, int(ncol)))
        for handles, title, ncol in legend_rows
        if list(handles)
    ]
    if not nonempty_rows:
        return

    legend_ax = fig.add_axes(bounds)
    legend_ax.axis("off")
    _legend_container_patch(legend_ax, x=0.01, y=0.10, width=0.98, height=0.80)
    n_rows = len(nonempty_rows)
    if row_title_y is None:
        row_title_y = [0.74, 0.34] if n_rows == 2 else list(np.linspace(0.78, 0.26, n_rows))
    if row_legend_y is None:
        row_legend_y = [0.60, 0.20] if n_rows == 2 else list(np.linspace(0.66, 0.14, n_rows))

    for idx, ((handles, title, ncol), title_y, legend_y) in enumerate(zip(nonempty_rows, row_title_y, row_legend_y)):
        if title:
            legend_ax.text(
                title_x,
                float(title_y),
                title,
                transform=legend_ax.transAxes,
                ha="left",
                va="top",
                fontsize=title_fontsize,
            )
        legend = legend_ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(legend_x, float(legend_y)),
            ncol=ncol,
            fontsize=entry_fontsize,
            frameon=False,
            borderaxespad=0.0,
            labelspacing=0.50,
            handlelength=1.9,
            handletextpad=0.55,
            columnspacing=1.1,
        )
        legend._legend_box.align = "left"
        if idx < len(nonempty_rows) - 1:
            legend_ax.add_artist(legend)


def place_panel_legends(
        fig,
        axes,
        used_axes: int,
        *,
        model_handles: Sequence[Line2D],
        extra_handles: Sequence[Line2D] = (),
        model_title: str = "Classifier regime",
        extra_title: str = "Encoding",
) -> None:
    """Place one grouped legend entity in the spare panel when available."""

    model_handles = list(model_handles)
    extra_handles = list(extra_handles)
    legend_blocks: List[Tuple[Sequence[Line2D], str | None, int]] = []
    if model_handles:
        legend_blocks.append((model_handles, model_title, 1))
    if extra_handles:
        legend_blocks.append((extra_handles, extra_title or None, 1))

    if used_axes < len(axes):
        _draw_grouped_legend_box(axes[used_axes], legend_blocks)
        finish_panel_grid(fig, axes, used_axes + 1)
        return

    if legend_blocks:
        _draw_horizontal_grouped_legend_box(
            fig,
            bounds=(0.11, 0.855, 0.78, 0.115),
            legend_rows=legend_blocks,
            title_x=0.04,
            legend_x=0.30,
            entry_fontsize=THESIS_LEGEND_SIZE - 0.2,
        )


def place_stacked_panel_legends(
        fig,
        axes,
        used_axes: int,
        *,
        legend_blocks: Sequence[Tuple[Sequence[Line2D], str | None, int]],
) -> None:
    """Place compact titled legend groups inside one boxed legend entity."""

    nonempty_blocks: List[Tuple[List[Line2D], str | None, int]] = [
        (list(handles), title, max(1, int(ncol)))
        for handles, title, ncol in legend_blocks
        if list(handles)
    ]
    if not nonempty_blocks:
        finish_panel_grid(fig, axes, used_axes)
        return

    if used_axes < len(axes):
        _draw_grouped_legend_box(
            axes[used_axes],
            nonempty_blocks,
            title_fontsize=THESIS_LEGEND_SIZE - 0.05,
            entry_fontsize=THESIS_LEGEND_SIZE - 0.35,
        )
        finish_panel_grid(fig, axes, used_axes + 1)
        return

    _draw_horizontal_grouped_legend_box(
        fig,
        bounds=(0.10, 0.84, 0.80, 0.14),
        legend_rows=nonempty_blocks,
        title_x=0.04,
        legend_x=0.31,
        title_fontsize=THESIS_LEGEND_SIZE,
        entry_fontsize=THESIS_LEGEND_SIZE - 0.2,
        row_title_y=list(np.linspace(0.78, 0.24, len(nonempty_blocks))),
        row_legend_y=list(np.linspace(0.66, 0.12, len(nonempty_blocks))),
    )

def tradeoff_regime_percent(label: str) -> int | None:
    text = str(label).strip()
    if not text.endswith("%"):
        return None
    try:
        return int(round(float(text.removesuffix("%"))))
    except ValueError:
        return None


def tradeoff_regime_linestyle(label: str) -> str:
    pct = tradeoff_regime_percent(label)
    if pct in TRADEOFF_REGIME_LINESTYLES:
        return TRADEOFF_REGIME_LINESTYLES[pct]
    return "-"


def classifier_regime_legend_handles(percentages: Sequence[int]) -> List[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=2.2,
            linestyle=TRADEOFF_REGIME_LINESTYLES.get(int(pct), "-"),
            label=f"{int(pct)}%",
        )
        for pct in percentages
    ]


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
    value = float(value)
    standard_labels = {
        0.0: "0",
        0.01: "0.01",
        0.05: "0.05",
        0.1: "0.10",
        0.25: "0.25",
    }
    for standard_value, label in standard_labels.items():
        if math.isclose(value, standard_value, rel_tol=0.0, abs_tol=1e-12):
            return label
    if abs(value) < 1e-12:
        return "0"
    if 0.005 <= value < 1.0:
        return f"{value:.2f}"
    if value >= 1.0:
        return f"{value:g}"
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
        rotate_labels = len(ticks) >= 4
        ax.set_xticklabels(
            [format_cutoff_axis_value(value) for value in ticks],
            rotation=40 if rotate_labels else 0,
            ha="right" if rotate_labels else "center",
            rotation_mode="anchor",
        )


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
        linestyle: str = "-",
        alpha: float = 0.98,
        show_band: bool = False,
        band_alpha: float = 0.14,
) -> None:
    if not points:
        return

    x = np.asarray([float(point["x"]) for point in points], dtype=float)
    y = np.asarray([float(point["y"]) for point in points], dtype=float)
    ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, label=label)

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
            markersize=13.5,
            label=f"Mass-targeted ({str(label).removeprefix('approx_mass_').replace('p', '.')})",
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
        style_legend_frame(legend)
        fig.add_artist(legend)


def add_exact_runtime_reference_line(
        ax,
        row_by_label: Dict[str, Dict[str, Any]],
        metric_key: str,
        *,
        color: Any,
        value_scale: float = 1.0,
) -> None:
    exact_row = row_by_label.get("exact")
    if exact_row is None:
        return
    value = finite_float_or_none(exact_row.get(metric_key))
    if value is None or value <= 0.0:
        return
    ax.axhline(
        value_scale * value,
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
) -> None:
    """Plot runtime and retained accuracy with base-model accuracy collapsed into line style.

    One panel is shown per term count. Speedup and retained accuracy are
    distinguished only by color. The three base-model classifier regimes are
    distinguished by line style (50% dotted, 70% dashed, 90% solid). The
    outer model trajectories also bound a light-to-dark gradient band used
    only as a readability aid; the legend therefore documents the actual
    semantic encodings (metric color and classifier-regime line style), not
    the gradient.
    """
    if not summary_rows:
        return

    model_ids = ordered_model_ids(summary_rows)
    term_counts = sorted({int(value) for value in term_counts})
    approx_thresholds = [
        str(label)
        for label in non_exact_threshold_labels(threshold_order)
        if any(str(row.get("threshold_label")) == str(label) for row in summary_rows)
    ]
    if not model_ids or not term_counts or not approx_thresholds:
        return

    model_styles = build_model_styles(summary_rows)
    speedup_color = TRADEOFF_COLORS["speedup"]
    accuracy_color = TRADEOFF_COLORS["accuracy"]
    baseline_color = TRADEOFF_COLORS["baseline"]

    # Preserve the familiar blue/orange metric colors. The gradient endpoints
    # are only a readability aid between the lowest/highest base-model traces.
    speedup_band_low = "#9ecae1"
    speedup_band_high = "#08519c"
    accuracy_band_low = "#fdae6b"
    accuracy_band_high = "#a63603"

    # Base-model classifier regime is encoded only by line style: 50%
    # dotted, 70% dashed, 90% solid.  No regime is visually privileged.
    model_line_styles = {
        str(model_id): tradeoff_regime_linestyle(str(model_styles[str(model_id)]["label"]))
        for model_id in model_ids
    }

    fixed_cutoffs = fixed_cutoff_tick_values(approx_thresholds)
    include_zero = any(abs(value) < 1e-12 for value in fixed_cutoffs)

    def gradient_fill_between(
            ax: Any,
            x_values: np.ndarray,
            lower_values: np.ndarray,
            upper_values: np.ndarray,
            color_low: str,
            color_high: str,
            *,
            alpha: float,
            steps: int = 28,
            zorder: float = 1.0,
    ) -> None:
        color0 = np.asarray(mcolors.to_rgb(color_low), dtype=float)
        color1 = np.asarray(mcolors.to_rgb(color_high), dtype=float)
        for step in range(steps):
            t0 = float(step) / float(steps)
            t1 = float(step + 1) / float(steps)
            band0 = lower_values + t0 * (upper_values - lower_values)
            band1 = lower_values + t1 * (upper_values - lower_values)
            midpoint = 0.5 * (t0 + t1)
            color = color0 * (1.0 - midpoint) + color1 * midpoint
            ax.fill_between(
                x_values,
                band0,
                band1,
                color=color,
                alpha=alpha,
                linewidth=0.0,
                zorder=zorder,
            )

    panel_series: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for model_id in model_ids:
        for n_terms in term_counts:
            rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
            row_by_label = {str(row["threshold_label"]): row for row in rows}
            exact_row = row_by_label.get("exact")
            exact_accuracy = finite_float_or_none(exact_row.get("accuracy")) if exact_row is not None else None
            if exact_accuracy is None or exact_accuracy <= 0.0:
                continue

            points: List[Dict[str, Any]] = []
            for order_idx, label in enumerate(approx_thresholds):
                row = row_by_label.get(str(label))
                if row is None:
                    continue
                cutoff_value = cutoff_x_value_for_row(row)
                speedup_value = speedup_vs_exact_from_row(row)
                accuracy_value = finite_float_or_none(row.get("accuracy"))
                if cutoff_value is None or speedup_value is None or accuracy_value is None:
                    continue
                points.append(
                    {
                        "threshold_label": str(label),
                        "cutoff": float(cutoff_value),
                        "x": float(cutoff_value),
                        "order_idx": int(order_idx),
                        "speedup": float(speedup_value),
                        "accuracy_retained": 100.0 * float(accuracy_value / exact_accuracy),
                        "adaptive": is_adaptive_threshold_row(row),
                    }
                )

            points.sort(key=lambda item: (float(item["cutoff"]), int(item["order_idx"])))
            if points:
                panel_series[(str(model_id), int(n_terms))] = {
                    "points": points,
                    "line_style": model_line_styles[str(model_id)],
                    "label": str(model_styles[str(model_id)]["label"]),
                }

    if not panel_series:
        return

    all_x_values = [
        float(point["x"])
        for series in panel_series.values()
        for point in series["points"]
        if math.isfinite(float(point["x"]))
    ]

    nrows, ncols = term_panel_grid(term_counts)
    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=thesis_panel_figure_size(nrows),
        squeeze=False,
        constrained_layout=False,
    )
    axes = list(axes_grid.flatten())

    for panel_idx, n_terms in enumerate(term_counts):
        ax = axes[panel_idx]
        ax_right = ax.twinx()

        ax.axhline(1.0, color=baseline_color, linestyle="--", linewidth=1.0, alpha=0.90, zorder=1)
        ax_right.axhline(100.0, color=baseline_color, linestyle="--", linewidth=1.0, alpha=0.90, zorder=1)

        available_models = [
            str(model_id)
            for model_id in model_ids
            if (str(model_id), int(n_terms)) in panel_series
        ]

        # Gradient band between the lowest/highest base-model traces. Adaptive
        # cutoffs differ by model, so interpolate each boundary trajectory onto
        # the union of their displayed x coordinates before filling.
        if len(available_models) >= 2:
            low_points = panel_series[(available_models[0], int(n_terms))]["points"]
            high_points = panel_series[(available_models[-1], int(n_terms))]["points"]
            low_x = np.asarray([float(point["x"]) for point in low_points], dtype=float)
            high_x = np.asarray([float(point["x"]) for point in high_points], dtype=float)
            band_x = np.asarray(sorted(set(low_x.tolist() + high_x.tolist())), dtype=float)

            low_speed = np.interp(
                band_x,
                low_x,
                np.asarray([float(point["speedup"]) for point in low_points], dtype=float),
            )
            high_speed = np.interp(
                band_x,
                high_x,
                np.asarray([float(point["speedup"]) for point in high_points], dtype=float),
            )
            gradient_fill_between(
                ax,
                band_x,
                low_speed,
                high_speed,
                speedup_band_low,
                speedup_band_high,
                alpha=0.18,
                zorder=1.2,
            )

            low_accuracy = np.interp(
                band_x,
                low_x,
                np.asarray([float(point["accuracy_retained"]) for point in low_points], dtype=float),
            )
            high_accuracy = np.interp(
                band_x,
                high_x,
                np.asarray([float(point["accuracy_retained"]) for point in high_points], dtype=float),
            )
            gradient_fill_between(
                ax_right,
                band_x,
                low_accuracy,
                high_accuracy,
                accuracy_band_low,
                accuracy_band_high,
                alpha=0.14,
                zorder=1.2,
            )

        for model_id in available_models:
            series = panel_series[(model_id, int(n_terms))]
            points = series["points"]
            line_style = str(series["line_style"])

            x_values = np.asarray([float(point["x"]) for point in points], dtype=float)
            speed_values = np.asarray([float(point["speedup"]) for point in points], dtype=float)
            accuracy_values = np.asarray([float(point["accuracy_retained"]) for point in points], dtype=float)

            ax.plot(
                x_values,
                speed_values,
                color=speedup_color,
                linestyle=line_style,
                linewidth=TRADEOFF_LINE_WIDTH,
                marker="o",
                markersize=TRADEOFF_MARKER_SIZE,
                zorder=2.6,
            )
            ax_right.plot(
                x_values,
                accuracy_values,
                color=accuracy_color,
                linestyle=line_style,
                linewidth=TRADEOFF_LINE_WIDTH,
                marker="o",
                markersize=TRADEOFF_MARKER_SIZE,
                zorder=2.6,
            )

            adaptive_points = [point for point in points if bool(point.get("adaptive", False))]
            for point in adaptive_points:
                star_size = TRADEOFF_ADAPTIVE_MARKER_SIZE
                ax.scatter(
                    [float(point["x"])],
                    [float(point["speedup"])],
                    marker="*",
                    s=star_size,
                    color=speedup_color,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=5,
                )
                ax_right.scatter(
                    [float(point["x"])],
                    [float(point["accuracy_retained"])],
                    marker="*",
                    s=star_size,
                    color=accuracy_color,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=5,
                )

        ax.set_title(f"{int(n_terms)} terms", fontsize=THESIS_PANEL_LABEL_SIZE, pad=6)
        configure_numeric_cutoff_axis(
            ax,
            tick_values=fixed_cutoffs,
            point_values=all_x_values,
            include_zero=include_zero,
        )
        ax.set_ylim(0.0, 2.0)
        ax_right.set_ylim(0.0, 200.0)
        ax.grid(axis="y", alpha=THESIS_GRID_ALPHA)
        ax.grid(axis="x", alpha=0.08)
        ax.set_axisbelow(True)

        show_left_axis = panel_idx % ncols == 0
        # Keep the secondary axis on the actual right column only.  Showing
        # it again on the bottom-left panel crowds the spare legend cell and
        # makes the 2x2 layout look misaligned.
        show_right_axis = panel_idx % ncols == ncols - 1

        ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
        if show_left_axis:
            ax.tick_params(axis="y", colors=speedup_color)
            ax.spines["left"].set_color(speedup_color)
            ax.spines["left"].set_linewidth(1.0)
            ax.set_yticklabels(["0", "0.5", "1.0", "1.5", "2.0"], color=speedup_color)
        else:
            ax.tick_params(axis="y", labelleft=False, left=False)
            ax.spines["left"].set_visible(False)

        ax_right.set_yticks([0.0, 50.0, 100.0, 150.0, 200.0])
        if show_right_axis:
            ax_right.tick_params(axis="y", colors=accuracy_color)
            ax_right.spines["right"].set_visible(True)
            ax_right.spines["right"].set_color(accuracy_color)
            ax_right.spines["right"].set_linewidth(1.0)
            ax_right.set_yticklabels(["0%", "50%", "100%", "150%", "200%"], color=accuracy_color)
        else:
            ax_right.tick_params(axis="y", labelright=False, right=False)
            ax_right.spines["right"].set_visible(False)

    metric_handles = [
        Line2D(
            [0],
            [0],
            color=speedup_color,
            linewidth=2.6,
            marker="o",
            markersize=4.8,
            label="Speedup",
        ),
        Line2D(
            [0],
            [0],
            color=accuracy_color,
            linewidth=2.4,
            marker="o",
            markersize=4.2,
            label="Retained accuracy",
        ),
    ]
    model_handles = classifier_regime_legend_handles(TRADEOFF_REGIME_ORDER)
    reference_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="*",
            markerfacecolor="#555555",
            markeredgecolor="white",
            linewidth=0.0,
            markersize=13.5,
            label="Mass-targeted (0.8)",
        ),
        Line2D(
            [0],
            [0],
            color=baseline_color,
            linestyle="--",
            linewidth=1.0,
            label="Exact reference",
        ),
    ]
    place_stacked_panel_legends(
        fig,
        axes,
        len(term_counts),
        legend_blocks=[
            (metric_handles, "Metric", 2),
            (model_handles, "Classifier regime", 3),
            (reference_handles, "Reference", 1),
        ],
    )
    fig.supxlabel("Path-mass cutoff", y=0.026, fontsize=THESIS_PANEL_LABEL_SIZE - 0.8)
    fig.text(0.028, 0.52, "Speedup", ha="center", va="center", rotation="vertical", color=speedup_color, fontsize=THESIS_PANEL_LABEL_SIZE - 0.8)
    fig.text(0.958, 0.52, "Retained accuracy", ha="center", va="center", rotation=-90, color=accuracy_color, fontsize=THESIS_PANEL_LABEL_SIZE - 0.8)
    fig.subplots_adjust(left=0.13, right=0.86, bottom=0.15, top=0.96, hspace=0.36, wspace=0.34)

    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_mnist_lookup_accuracy_tradeoff(
        summary_rows: List[Dict[str, Any]],
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
        output_path: Path,
) -> None:
    """Plot exact-normalized MNIST lookup cost against retained accuracy.

    The three standard base-model accuracy levels are collapsed into each term
    panel.  Metric identity is encoded by color (blue lookup ratio, orange
    retained accuracy), while base-model accuracy is encoded consistently by
    line style (50% dotted, 70% dashed, 90% solid).  The translucent gradients
    between the 50% and 90% fixed-cutoff trajectories are only a readability
    aid; the actual model identity remains the line style.

    Mass-targeted points are positioned at their searched numerical
    cutoff and are connected into their corresponding model trajectory.  They
    are excluded from the gradient band because the searched cutoff can differ
    across base models.
    """
    if not summary_rows:
        return

    model_ids = ordered_model_ids(summary_rows)
    if not model_ids or not term_counts:
        return

    # This thesis-facing plot is intentionally specialized to the three
    # standard base-model accuracy levels used by Pipeline I.
    model_by_pct: Dict[int, str] = {}
    for model_id in model_ids:
        label = str(build_model_styles(summary_rows)[model_id]["label"])
        for pct in (50, 70, 90):
            if label.startswith(f"{pct}%"):
                model_by_pct[pct] = model_id
                break
    if any(pct not in model_by_pct for pct in (50, 70, 90)):
        return

    approx_thresholds = [
        label
        for label in non_exact_threshold_labels(threshold_order)
        if any(str(row.get("threshold_label")) == label for row in summary_rows)
    ]
    if not approx_thresholds:
        return

    fixed_values = fixed_cutoff_tick_values(approx_thresholds)
    if not fixed_values:
        return
    if not any(abs(value) < 1e-12 for value in fixed_values):
        fixed_values = [0.0, *fixed_values]
    fixed_values = sorted({round(float(value), 12) for value in fixed_values})
    include_zero = any(abs(value) < 1e-12 for value in fixed_values)

    lookup_color = TRADEOFF_COLORS["speedup"]
    accuracy_color = TRADEOFF_COLORS["accuracy"]
    baseline_color = TRADEOFF_COLORS["baseline"]
    positive_zone_color = TRADEOFF_COLORS["positive_zone"]

    # Same visual family as the accepted runtime–accuracy template.
    lookup_band_low = "#9ecae1"
    lookup_band_high = "#08519c"
    accuracy_band_low = "#fdae6b"
    accuracy_band_high = "#a63603"
    model_linestyles = dict(TRADEOFF_REGIME_LINESTYLES)
    adaptive_sizes = {pct: TRADEOFF_ADAPTIVE_MARKER_SIZE for pct in TRADEOFF_REGIME_ORDER}

    def gradient_band(
            ax: Any,
            x_values: Sequence[float],
            start_values: Sequence[float],
            end_values: Sequence[float],
            start_color: str,
            end_color: str,
            *,
            alpha: float,
            steps: int = 28,
    ) -> None:
        x_array = np.asarray(x_values, dtype=float)
        start_array = np.asarray(start_values, dtype=float)
        end_array = np.asarray(end_values, dtype=float)
        if x_array.size < 2 or start_array.size != x_array.size or end_array.size != x_array.size:
            return
        color_start = np.asarray(mcolors.to_rgb(start_color), dtype=float)
        color_end = np.asarray(mcolors.to_rgb(end_color), dtype=float)
        for step in range(steps):
            t0 = step / steps
            t1 = (step + 1) / steps
            lower = start_array + t0 * (end_array - start_array)
            upper = start_array + t1 * (end_array - start_array)
            midpoint = 0.5 * (t0 + t1)
            color = (1.0 - midpoint) * color_start + midpoint * color_end
            ax.fill_between(
                x_array,
                lower,
                upper,
                color=color,
                alpha=alpha,
                linewidth=0.0,
                zorder=1,
            )

    def series_for(model_id: str, n_terms: int) -> List[Dict[str, Any]]:
        rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
        row_by_label = {str(row["threshold_label"]): row for row in rows}
        exact_row = row_by_label.get("exact")
        if exact_row is None:
            return []
        exact_lookups = finite_float_or_none(exact_row.get("mean_read_mnist_lookup_calls"))
        exact_accuracy = finite_float_or_none(exact_row.get("accuracy"))
        if exact_lookups is None or exact_lookups <= 0.0 or exact_accuracy is None or exact_accuracy <= 0.0:
            return []

        points: List[Dict[str, Any]] = []
        for order_idx, label in enumerate(approx_thresholds):
            row = row_by_label.get(label)
            if row is None:
                continue
            cutoff_value = cutoff_x_value_for_row(row)
            lookup_value = finite_float_or_none(row.get("mean_read_mnist_lookup_calls"))
            accuracy_value = finite_float_or_none(row.get("accuracy"))
            if cutoff_value is None or lookup_value is None or accuracy_value is None:
                continue
            points.append(
                {
                    "label": str(label),
                    "cutoff": float(cutoff_value),
                    "x": float(cutoff_value),
                    "lookup_ratio": float(lookup_value / exact_lookups),
                    "accuracy_retained": float(100.0 * accuracy_value / exact_accuracy),
                    "adaptive": is_adaptive_threshold_row(row),
                    "order_idx": int(order_idx),
                }
            )
        points.sort(key=lambda item: (float(item["cutoff"]), int(item["order_idx"])))
        return points

    nrows, ncols = term_panel_grid(term_counts)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=thesis_panel_figure_size(nrows),
        squeeze=False,
        constrained_layout=False,
        sharey=True,
    )
    panel_axes = list(axes.ravel())

    all_x_values = [
        float(value)
        for row in summary_rows
        if str(row.get("threshold_label")) in approx_thresholds
        and (value := cutoff_x_value_for_row(row)) is not None
    ]

    right_axes: List[Any] = []
    for panel_idx, (ax, n_terms) in enumerate(zip(panel_axes, term_counts)):
        ax_right = ax.twinx()
        right_axes.append(ax_right)

        # Lookup ratio is inverted so fewer lookups appear upward.  The upper
        # half therefore represents an improvement over the exact ratio of 1.
        ax.axhspan(0.0, 1.0, color=positive_zone_color, alpha=0.055, zorder=0)
        ax.axhline(1.0, color=baseline_color, linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)
        ax_right.axhline(100.0, color=baseline_color, linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)

        panel_series = {
            pct: series_for(model_by_pct[pct], int(n_terms))
            for pct in (50, 70, 90)
        }

        # Readability gradient uses fixed cutoffs only.  Adaptive mass points
        # have model-specific searched cutoffs and therefore should not define
        # a common envelope.
        fixed_band_rows: Dict[int, Dict[float, Dict[str, Any]]] = {}
        for pct in (50, 90):
            fixed_band_rows[pct] = {
                round(float(point["cutoff"]), 12): point
                for point in panel_series[pct]
                if not bool(point["adaptive"])
            }
        common_fixed = sorted(set(fixed_band_rows[50]) & set(fixed_band_rows[90]))
        if len(common_fixed) >= 2:
            band_x = list(common_fixed)
            gradient_band(
                ax,
                band_x,
                [fixed_band_rows[50][value]["lookup_ratio"] for value in common_fixed],
                [fixed_band_rows[90][value]["lookup_ratio"] for value in common_fixed],
                lookup_band_low,
                lookup_band_high,
                alpha=0.18,
            )
            gradient_band(
                ax_right,
                band_x,
                [fixed_band_rows[50][value]["accuracy_retained"] for value in common_fixed],
                [fixed_band_rows[90][value]["accuracy_retained"] for value in common_fixed],
                accuracy_band_low,
                accuracy_band_high,
                alpha=0.14,
            )

        for pct in (50, 70, 90):
            points = panel_series[pct]
            if not points:
                continue
            x_values = [float(point["x"]) for point in points]
            lookup_values = [float(point["lookup_ratio"]) for point in points]
            accuracy_values = [float(point["accuracy_retained"]) for point in points]
            linestyle = model_linestyles[pct]

            ax.plot(
                x_values,
                lookup_values,
                color=lookup_color,
                linewidth=TRADEOFF_LINE_WIDTH,
                linestyle=linestyle,
                marker="o",
                markersize=TRADEOFF_MARKER_SIZE,
                zorder=2.6,
            )
            ax_right.plot(
                x_values,
                accuracy_values,
                color=accuracy_color,
                linewidth=TRADEOFF_LINE_WIDTH,
                linestyle=linestyle,
                marker="o",
                markersize=TRADEOFF_MARKER_SIZE,
                zorder=2.6,
            )

            adaptive_points = [point for point in points if bool(point["adaptive"])]
            if adaptive_points:
                ax.scatter(
                    [float(point["x"]) for point in adaptive_points],
                    [float(point["lookup_ratio"]) for point in adaptive_points],
                    marker="*",
                    s=adaptive_sizes[pct],
                    color=lookup_color,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=5,
                )
                ax_right.scatter(
                    [float(point["x"]) for point in adaptive_points],
                    [float(point["accuracy_retained"]) for point in adaptive_points],
                    marker="*",
                    s=adaptive_sizes[pct],
                    color=accuracy_color,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=5,
                )

        ax.set_title(f"{int(n_terms)} terms", fontsize=THESIS_PANEL_LABEL_SIZE, pad=6)
        configure_numeric_cutoff_axis(
            ax,
            tick_values=fixed_values,
            point_values=all_x_values,
            include_zero=include_zero,
        )
        ax.set_ylim(2.0, 0.0)
        ax_right.set_ylim(0.0, 200.0)
        ax.grid(axis="y", alpha=THESIS_GRID_ALPHA)
        ax.grid(axis="x", alpha=0.10)
        ax.set_axisbelow(True)

        show_left_axis = panel_idx % ncols == 0
        # Keep the secondary axis on the actual right column only.  Showing
        # it again on the bottom-left panel crowds the spare legend cell and
        # makes the 2x2 layout look misaligned.
        show_right_axis = panel_idx % ncols == ncols - 1

        ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
        if show_left_axis:
            ax.tick_params(axis="y", colors=lookup_color)
            ax.spines["left"].set_color(lookup_color)
            ax.spines["left"].set_linewidth(1.0)
            ax.set_yticklabels(["0", "0.5", "1.0", "1.5", "2.0"], color=lookup_color)
        else:
            ax.tick_params(axis="y", labelleft=False, left=False)
            ax.spines["left"].set_visible(False)

        ax_right.set_yticks([0.0, 50.0, 100.0, 150.0, 200.0])
        if show_right_axis:
            ax_right.tick_params(axis="y", colors=accuracy_color)
            ax_right.spines["right"].set_visible(True)
            ax_right.spines["right"].set_color(accuracy_color)
            ax_right.spines["right"].set_linewidth(1.0)
            ax_right.set_yticklabels(["0%", "50%", "100%", "150%", "200%"], color=accuracy_color)
        else:
            ax_right.tick_params(axis="y", labelright=False, right=False)
            ax_right.spines["right"].set_visible(False)

    metric_handles = [
        Line2D(
            [0],
            [0],
            color=lookup_color,
            linewidth=2.6,
            marker="o",
            markersize=4.8,
            label="Lookup ratio",
        ),
        Line2D(
            [0],
            [0],
            color=accuracy_color,
            linewidth=2.4,
            marker="o",
            markersize=4.2,
            label="Retained accuracy",
        ),
    ]
    model_handles = classifier_regime_legend_handles(TRADEOFF_REGIME_ORDER)
    reference_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=0.0,
            marker="*",
            markersize=13.5,
            label="Mass-targeted (0.8)",
        ),
        Line2D([0], [0], color=baseline_color, linewidth=1.0, linestyle="--", label="Exact reference"),
    ]
    place_stacked_panel_legends(
        fig,
        panel_axes,
        len(term_counts),
        legend_blocks=[
            (metric_handles, "Metric", 2),
            (model_handles, "Classifier regime", 3),
            (reference_handles, "Reference", 1),
        ],
    )

    fig.supxlabel("Path-mass cutoff", y=0.026, fontsize=THESIS_PANEL_LABEL_SIZE - 0.8)
    fig.text(0.028, 0.52, "Lookup ratio", ha="center", va="center", rotation="vertical", color=lookup_color, fontsize=THESIS_PANEL_LABEL_SIZE - 0.8)
    fig.text(0.958, 0.52, "Retained accuracy", ha="center", va="center", rotation=-90, color=accuracy_color, fontsize=THESIS_PANEL_LABEL_SIZE - 0.8)
    fig.subplots_adjust(left=0.13, right=0.86, bottom=0.15, top=0.96, hspace=0.36, wspace=0.34)

    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_true_candidate_metric_vs_cutoff(
        summary_rows: List[Dict[str, Any]],
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
        output_path: Path,
        *,
        metric_key: str,
        ylabel: str,
        yscale: str | None = None,
        ylim: Tuple[float, float] | None = None,
        shared_ylabel: bool = False,
) -> None:
    if not summary_rows or not metric_has_usable_values(summary_rows, metric_key):
        output_path.unlink(missing_ok=True)
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
        figsize=thesis_panel_figure_size(nrows),
        squeeze=False,
        constrained_layout=False,
    )
    axes = list(axes_grid.flatten())

    for panel_idx, (ax, n_terms) in enumerate(zip(axes, term_counts)):
        for model_id in model_ids:
            rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
            row_by_label = {str(row["threshold_label"]): row for row in rows}
            runtime_scale = 1000.0 if metric_key == "median_true_candidate_runtime_sec" else 1.0
            points = cutoff_series_points(
                row_by_label,
                approx_thresholds,
                metric_key,
                value_scale=runtime_scale,
            )
            color = model_styles[model_id]["color"]
            if metric_key == "median_true_candidate_runtime_sec":
                add_exact_runtime_reference_line(
                    ax,
                    row_by_label,
                    metric_key,
                    color=color,
                    value_scale=runtime_scale,
                )
            plot_cutoff_metric_series(
                ax,
                points,
                color=color,
                label=model_styles[model_id]["label"],
            )
        ax.set_title(f"{int(n_terms)} terms", fontsize=THESIS_PANEL_LABEL_SIZE, pad=6)
        configure_numeric_cutoff_axis(
            ax,
            tick_values=tick_values,
            point_values=all_x_values,
            include_zero=include_zero,
        )
        # Figure-level axis titles are added below; keep numeric ticks per panel.
        if yscale is not None:
            ax.set_yscale(yscale)
        if metric_key == "median_true_candidate_runtime_sec":
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=THESIS_GRID_ALPHA)

    handles = model_line_handles(model_ids, model_styles)
    extra_handles: List[Line2D] = []
    if metric_key == "median_true_candidate_runtime_sec":
        extra_handles.append(exact_reference_handle("Exact runtime"))
    extra_handles.extend(adaptive_marker_handles(approx_thresholds))
    place_panel_legends(
        fig,
        axes,
        len(term_counts),
        model_handles=handles,
        extra_handles=extra_handles,
        extra_title="Reference",
    )
    fig.supxlabel("Path-mass cutoff", y=0.018)
    fig.supylabel(ylabel, x=0.018)
    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.14,
        top=0.96,
        hspace=0.42,
        wspace=0.30,
    )
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.08)
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
        figsize=thesis_panel_figure_size(nrows),
        squeeze=False,
        constrained_layout=False,
    )
    axes = list(axes_grid.flatten())

    for panel_idx, (ax, n_terms) in enumerate(zip(axes, term_counts)):
        for model_id in model_ids:
            rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
            row_by_label = {str(row["threshold_label"]): row for row in rows}
            points = cutoff_series_points(
                row_by_label,
                approx_thresholds,
                "median_runtime_sec",
                lower_key="runtime_q25_sec",
                upper_key="runtime_q75_sec",
                value_scale=1000.0,
            )
            color = model_styles[model_id]["color"]
            add_exact_runtime_reference_line(
                ax,
                row_by_label,
                "median_runtime_sec",
                color=color,
                value_scale=1000.0,
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
        ax.set_title(f"{int(n_terms)} terms", fontsize=THESIS_PANEL_LABEL_SIZE, pad=6)
        configure_numeric_cutoff_axis(
            ax,
            tick_values=tick_values,
            point_values=all_x_values,
            include_zero=include_zero,
        )
        # Figure-level axis titles are added below; keep numeric ticks per panel.
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=THESIS_GRID_ALPHA)

    model_handles = model_line_handles(model_ids, model_styles)
    extra_handles = [
        exact_reference_handle("Exact runtime"),
        *adaptive_marker_handles(approx_thresholds),
    ]
    place_panel_legends(
        fig,
        axes,
        len(term_counts),
        model_handles=model_handles,
        extra_handles=extra_handles,
        extra_title="Reference",
    )
    fig.supxlabel("Path-mass cutoff", y=0.018)
    fig.supylabel("Median runtime (ms)", x=0.018)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.96, hspace=0.42, wspace=0.30)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_true_candidate_metric_row(
        summary_rows: List[Dict[str, Any]],
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
        output_path: Path,
        *,
        metric_key: str,
        ylabel: str,
        ylim: Tuple[float, float] | None = None,
) -> None:
    """Compact appendix layout with one row of panels plus a dedicated legend cell."""

    if not summary_rows or not term_counts:
        return

    model_styles = build_model_styles(summary_rows)
    model_ids = ordered_model_ids(summary_rows)
    approx_thresholds = [
        str(label)
        for label in non_exact_threshold_labels(threshold_order)
        if any(str(row.get("threshold_label")) == str(label) for row in summary_rows)
    ]
    if not model_ids or not approx_thresholds:
        return

    tick_values = fixed_cutoff_tick_values(approx_thresholds)
    include_zero = any(abs(value) < 1e-12 for value in tick_values)
    all_x_values = [
        value
        for row in summary_rows
        if str(row.get("threshold_label")) in approx_thresholds
        and (value := cutoff_x_value_for_row(row)) is not None
    ]

    term_counts = sorted({int(value) for value in term_counts})
    width_ratios = [1.0 for _ in term_counts] + [0.92]
    fig, axes_grid = plt.subplots(
        1,
        len(term_counts) + 1,
        figsize=(THESIS_TEXT_WIDTH_IN, 2.85),
        squeeze=False,
        constrained_layout=False,
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes = list(axes_grid.ravel())
    panel_axes = axes[:len(term_counts)]
    legend_ax = axes[len(term_counts)]

    for panel_idx, (ax, n_terms) in enumerate(zip(panel_axes, term_counts)):
        for model_id in model_ids:
            rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
            row_by_label = {str(row["threshold_label"]): row for row in rows}
            style = model_styles[model_id]
            plot_cutoff_metric_series(
                ax,
                cutoff_series_points(row_by_label, approx_thresholds, metric_key),
                color=style["color"],
                linestyle=tradeoff_regime_linestyle(style["label"]),
                linewidth=1.9,
                show_band=False,
            )

        ax.set_title(f"{int(n_terms)} terms", fontsize=THESIS_PANEL_LABEL_SIZE - 1.0, pad=6)
        configure_numeric_cutoff_axis(
            ax,
            tick_values=tick_values,
            point_values=all_x_values,
            include_zero=include_zero,
        )
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=THESIS_GRID_ALPHA)
        if panel_idx > 0:
            ax.tick_params(axis="y", labelleft=False, left=False)
            ax.spines["left"].set_visible(False)

    classifier_handles = [
        Line2D(
            [0],
            [0],
            color=model_styles[model_id]["color"],
            linestyle=tradeoff_regime_linestyle(model_styles[model_id]["label"]),
            marker="o",
            markersize=4.6,
            linewidth=1.9,
            label=model_styles[model_id]["label"],
        )
        for model_id in model_ids
    ]
    reference_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color=TRADEOFF_COLORS["mass_marker"],
            markerfacecolor=TRADEOFF_COLORS["mass_marker"],
            markeredgecolor="white",
            linewidth=0,
            markersize=13.5,
            label="Mass-targeted (0.8)",
        ),
    ]
    _draw_grouped_legend_box(
        legend_ax,
        [
            (classifier_handles, "Classifier regime", 1),
            (reference_handles, "Reference", 1),
        ],
        title_fontsize=THESIS_LEGEND_SIZE - 0.05,
        entry_fontsize=THESIS_LEGEND_SIZE - 0.25,
    )
    fig.supxlabel("Path-mass cutoff", y=0.06, fontsize=THESIS_PANEL_LABEL_SIZE - 0.8)
    fig.supylabel(ylabel, x=0.035, fontsize=THESIS_PANEL_LABEL_SIZE - 0.8)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.90, bottom=0.22, wspace=0.26)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.06)
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
    min_delta = min(all_bounds, default=-5.0)
    y_lower = min(-5.0, 1.10 * min_delta if min_delta < 0.0 else -5.0)
    ylim = (y_lower, 5.0)

    nrows, ncols = term_panel_grid(term_counts)
    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=thesis_panel_figure_size(nrows),
        squeeze=False,
        constrained_layout=False,
    )
    axes = list(axes_grid.flatten())

    for panel_idx, (ax, n_terms) in enumerate(zip(axes, term_counts)):
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
        configure_numeric_cutoff_axis(
            ax,
            tick_values=tick_values,
            point_values=all_x_values,
            include_zero=False,
        )
        ax.set_title(f"{int(n_terms)} terms", fontsize=THESIS_PANEL_LABEL_SIZE, pad=6)
        ax.set_ylim(*ylim)
        # Figure-level axis titles are added below; keep numeric ticks per panel.
        ax.grid(alpha=THESIS_GRID_ALPHA)

    model_handles = model_line_handles(model_ids, model_styles)
    extra_handles = [
        exact_reference_handle("Exact baseline"),
        Line2D(
            [0],
            [0],
            marker="*",
            color=TRADEOFF_COLORS["mass_marker"],
            markerfacecolor=TRADEOFF_COLORS["mass_marker"],
            markeredgecolor="white",
            linewidth=0,
            markersize=13.5,
            label="Mass-targeted (0.8)",
        ),
    ]
    place_panel_legends(
        fig,
        axes,
        len(term_counts),
        model_handles=model_handles,
        extra_handles=extra_handles,
        extra_title="Reference",
    )
    fig.supxlabel("Path-mass cutoff", y=0.018)
    fig.supylabel("Accuracy change vs exact (pp)", x=0.018)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.96, hspace=0.42, wspace=0.30)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)



def plot_predictive_effects_combined(
        summary_rows: List[Dict[str, Any]],
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
        output_path: Path,
) -> None:
    """Combine accuracy change and true-sum survival into one 3x2 thesis figure."""

    if not summary_rows:
        return
    model_styles = build_model_styles(summary_rows)
    model_ids = ordered_model_ids(summary_rows)
    approx_thresholds = [
        str(label)
        for label in non_exact_threshold_labels(threshold_order)
        if any(str(row.get("threshold_label")) == str(label) for row in summary_rows)
    ]
    if not model_ids or not approx_thresholds:
        return

    term_counts = sorted({int(value) for value in term_counts})
    tick_values = fixed_cutoff_tick_values(approx_thresholds)
    include_zero = any(abs(value) < 1e-12 for value in tick_values)
    all_x_values = [
        value
        for row in summary_rows
        if str(row.get("threshold_label")) in approx_thresholds
        and (value := cutoff_x_value_for_row(row)) is not None
    ]

    all_accuracy_bounds: List[float] = []
    for row in summary_rows:
        if str(row.get("threshold_label")) not in approx_thresholds:
            continue
        for key in (
            "accuracy_delta_ci_lower_vs_exact",
            "accuracy_delta_ci_upper_vs_exact",
            "accuracy_delta_vs_exact",
        ):
            value = finite_float_or_none(row.get(key))
            if value is not None:
                all_accuracy_bounds.append(100.0 * float(value))
    min_delta = min(all_accuracy_bounds, default=-5.0)
    accuracy_ylim = (min(-5.0, 1.10 * min_delta if min_delta < 0.0 else -5.0), 5.0)

    nrows = len(term_counts)
    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(THESIS_TEXT_WIDTH_IN, max(5.65, 1.72 * nrows + 0.95)),
        squeeze=False,
        constrained_layout=False,
        sharex=False,
    )

    baseline_color = TRADEOFF_COLORS["baseline"]
    column_title_size = THESIS_PANEL_LABEL_SIZE - 1.2
    row_label_size = THESIS_PANEL_LABEL_SIZE - 1.9
    for row_idx, n_terms in enumerate(term_counts):
        ax_accuracy = axes_grid[row_idx][0]
        ax_survival = axes_grid[row_idx][1]
        ax_accuracy.axhline(0.0, color=baseline_color, linestyle="--", linewidth=1.0, alpha=0.85, zorder=1)
        ax_survival.axhline(1.0, color=baseline_color, linestyle="--", linewidth=1.0, alpha=0.85, zorder=1)

        for model_id in model_ids:
            rows = sorted_group_rows(get_rows(summary_rows, model_id, n_terms), threshold_order)
            row_by_label = {str(row["threshold_label"]): row for row in rows}
            color = model_styles[model_id]["color"]

            accuracy_points = cutoff_series_points(
                row_by_label,
                approx_thresholds,
                "accuracy_delta_vs_exact",
                lower_key="accuracy_delta_ci_lower_vs_exact",
                upper_key="accuracy_delta_ci_upper_vs_exact",
                value_scale=100.0,
            )
            plot_cutoff_metric_series(
                ax_accuracy,
                accuracy_points,
                color=color,
                linewidth=1.9,
                show_band=True,
                band_alpha=0.12,
            )

            survival_points = cutoff_series_points(
                row_by_label,
                approx_thresholds,
                "true_candidate_survival_rate",
            )
            plot_cutoff_metric_series(
                ax_survival,
                survival_points,
                color=color,
                linewidth=1.9,
                show_band=False,
            )

        for ax in (ax_accuracy, ax_survival):
            configure_numeric_cutoff_axis(
                ax,
                tick_values=tick_values,
                point_values=all_x_values,
                include_zero=include_zero,
            )
            ax.grid(alpha=THESIS_GRID_ALPHA)
            if row_idx < nrows - 1:
                ax.tick_params(axis="x", labelbottom=False)

        ax_accuracy.set_ylim(*accuracy_ylim)
        ax_survival.set_ylim(0.0, 1.05)
        ax_accuracy.text(
            0.03,
            0.93,
            f"{int(n_terms)} terms",
            transform=ax_accuracy.transAxes,
            ha="left",
            va="top",
            fontsize=row_label_size,
        )
        if row_idx == 0:
            ax_accuracy.set_title("Accuracy change vs exact (pp)", fontsize=column_title_size, pad=7)
            ax_survival.set_title("True-sum survival fraction", fontsize=column_title_size, pad=7)

    classifier_handles = model_line_handles(model_ids, model_styles)
    reference_handles = [
        Line2D([0], [0], color=baseline_color, linestyle="--", linewidth=1.0, label="Exact reference"),
        Line2D(
            [0],
            [0],
            marker="*",
            color=TRADEOFF_COLORS["mass_marker"],
            markerfacecolor=TRADEOFF_COLORS["mass_marker"],
            markeredgecolor="white",
            linewidth=0,
            markersize=13.5,
            label="Mass-targeted (0.8)",
        ),
    ]
    _draw_horizontal_grouped_legend_box(
        fig,
        bounds=(0.11, 0.86, 0.85, 0.085),
        legend_rows=[
            (classifier_handles, "Classifier regime", min(3, len(classifier_handles))),
            (reference_handles, "Reference", 2),
        ],
        title_x=0.04,
        legend_x=0.30,
        title_fontsize=THESIS_LEGEND_SIZE - 0.1,
        entry_fontsize=THESIS_LEGEND_SIZE - 0.15,
    )

    fig.supxlabel("Path-mass cutoff", y=0.026, fontsize=THESIS_PANEL_LABEL_SIZE - 0.8)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.09, top=0.82, hspace=0.22, wspace=0.24)
    ensure_parent = output_path.parent
    ensure_parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.08)
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
    ax.grid(axis="y", alpha=0.45)
    y_min = min(float(np.min(target)), float(np.min(achieved))) - 3.0
    y_max = max(float(np.max(target)), float(np.max(achieved))) + 3.0
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.6, len(model_rows) - 0.4)
    style_legend_frame(
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
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=thesis_panel_figure_size(nrows), squeeze=False)
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
            label = str(style.get("label", model_id))
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
        ax.set_title(f"{int(n_terms)} terms", fontsize=THESIS_PANEL_LABEL_SIZE, pad=6)
        ax.set_xlabel("Cutoff-search iteration")
        if ax_idx % ncols == 0:
            ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=THESIS_GRID_ALPHA)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    finish_panel_grid(fig, axes, len(term_counts))
    if not plotted_any:
        plt.close(fig)
        return
    reference_handles: List[Line2D] = []
    if value_key == "candidate_cutoff":
        reference_handles.append(
            Line2D([0], [0], color="#555555", marker="X", linewidth=0, markersize=7, label="Selected cutoff")
        )
    if show_target:
        reference_handles.append(
            Line2D([0], [0], color="#777777", linestyle="--", linewidth=1.0, label="Retained-mass target")
        )

    if len(term_counts) < len(axes):
        _draw_grouped_legend_box(
            axes[len(term_counts)],
            [
                (list(legend_handles.values()), "Classifier regime", 1),
                (reference_handles, "Reference", 1),
            ],
            title_fontsize=THESIS_LEGEND_SIZE - 0.05,
            entry_fontsize=THESIS_LEGEND_SIZE - 0.3,
        )
        finish_panel_grid(fig, axes, len(term_counts) + 1)
    elif legend_handles or reference_handles:
        _draw_horizontal_grouped_legend_box(
            fig,
            bounds=(0.11, 0.855, 0.78, 0.115),
            legend_rows=[
                (list(legend_handles.values()), "Classifier regime", min(4, len(legend_handles))),
                (reference_handles, "Reference", min(4, max(1, len(reference_handles)))),
            ],
            title_x=0.04,
            legend_x=0.30,
            entry_fontsize=THESIS_LEGEND_SIZE - 0.2,
        )

    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.96, hspace=0.42, wspace=0.30)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.12)
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

    fig.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.945, hspace=0.28, wspace=0.18)
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



def combine_vertical_figure_assets(
        input_paths: Sequence[Path],
        output_path: Path,
        *,
        panel_labels: Sequence[str],
        height_ratios: Sequence[float] | None = None,
        figure_height: float = 7.35,
) -> None:
    """Create a reproducible compact appendix plate from generated plot assets."""

    if len(input_paths) != len(panel_labels) or not input_paths:
        raise ValueError("input_paths and panel_labels must have the same non-zero length")
    if any(not Path(path).exists() for path in input_paths):
        return
    if height_ratios is None:
        height_ratios = [1.0 for _ in input_paths]
    fig = plt.figure(figsize=(THESIS_TEXT_WIDTH_IN, float(figure_height)), constrained_layout=False)
    grid = fig.add_gridspec(
        len(input_paths),
        1,
        height_ratios=list(height_ratios),
        hspace=0.10,
    )
    for idx, (path, label) in enumerate(zip(input_paths, panel_labels)):
        ax = fig.add_subplot(grid[idx, 0])
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_axis_off()
        ax.text(
            0.002,
            1.002,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=THESIS_PANEL_LABEL_SIZE,
            fontweight="regular",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.985, bottom=0.005)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_bundle_readme(
        path: Path,
        term_counts: Sequence[int],
        threshold_order: Sequence[str],
        *,
        include_biased_tradeoff: bool,
) -> None:
    figure_root = path.parent

    def exists(relative_path: str) -> bool:
        return (figure_root / relative_path).exists()

    main_entries = [
        ("main_text/runtime_accuracy_tradeoff_by_terms.png", "main thesis-facing runtime--accuracy tradeoff; standard models"),
        ("main_text/mnist_lookup_accuracy_tradeoff_by_terms.png", "relative digit-probability lookup count vs retained accuracy; standard models"),
        ("main_text/runtime_vs_cutoff_by_terms.png", "full-posterior runtime vs path-mass cutoff"),
        ("main_text/overhead_exact_vs_zero_cutoff_by_terms.png", "zero-cutoff control"),
        ("main_text/accuracy_delta_vs_exact_by_terms.png", "sum-accuracy change vs exact inference"),
        ("main_text/predictive_effects_by_terms.png", "combined accuracy-change and true-sum-survival figure"),
        ("main_text/true_candidate_runtime_vs_cutoff_by_terms.png", "true-sum-only runtime vs cutoff"),
        ("main_text/true_candidate_branch_count_vs_cutoff_by_terms.png", "true-sum branch count vs cutoff; emitted only when instrumentation is available"),
        ("main_text/true_candidate_survival_vs_cutoff_by_terms.png", "true-sum survival vs cutoff"),
        ("main_text/true_candidate_probability_vs_cutoff_by_terms.png", "normalized true-sum probability vs cutoff"),
    ]
    if include_biased_tradeoff:
        main_entries.insert(1, (
            "main_text/runtime_accuracy_tradeoff_biased_models_by_terms.png",
            "same runtime--accuracy design for the label-frequency robustness models",
        ))

    appendix_entries = [
        ("appendix/additional_inference_diagnostics_combined.png", "combined skewed-label tradeoff and normalized true-sum probability diagnostics"),
        ("appendix/target_vs_achieved_model_accuracy.png", "classifier-regime target vs achieved validation accuracy"),
        ("appendix/adaptive_topk_search_cutoff_iterations_by_terms.png", "mass-target cutoff-search trajectory"),
        ("appendix/adaptive_topk_search_mass_iterations_by_terms.png", "retained-mass trajectory during cutoff search"),
        ("appendix/heatmaps/heatmap_accuracy_by_model.png", "sum accuracy"),
        ("appendix/heatmaps/heatmap_output_pool_by_model.png", "surviving output-pool fraction"),
        ("appendix/heatmaps/heatmap_speedup_by_model.png", "speedup vs exact inference"),
        ("appendix/heatmaps/heatmap_true_candidate_probability_by_model.png", "normalized true-sum probability"),
        ("appendix/heatmaps/heatmap_true_candidate_survival_by_model.png", "true-sum survival"),
        ("appendix/heatmaps/heatmap_true_candidate_branch_count_by_model.png", "true-sum branch count; emitted only when instrumentation is available"),
        ("appendix/heatmaps/heatmap_true_candidate_speedup_by_model.png", "true-sum-only speedup"),
        ("appendix/heatmaps/heatmap_branch_count_by_model.png", "total branch count; emitted only when instrumentation is available"),
        ("appendix/heatmaps/heatmap_collapse_rate_by_model.png", "zero-mass posterior rate"),
    ]

    lines = [
        "Visualization bundle generated by visualize_results.py",
        "",
        "Main-text / primary outputs:",
    ]
    for relative_path, description in main_entries:
        if exists(relative_path):
            lines.append(f"- {relative_path}  [{description}]")

    lines.extend([
        "- tradeoff_alternatives/01_tradeoff_matrix_by_terms.png  [supporting alternative view]",
        "",
        "Appendix / supporting outputs:",
    ])
    for relative_path, description in appendix_entries:
        if exists(relative_path):
            lines.append(f"- {relative_path}  [{description}]")

    lines.extend([
        "",
        "Table outputs (under the visualization tables directory):",
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
