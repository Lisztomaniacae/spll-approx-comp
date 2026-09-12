from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from plot_palette import (
    FIGURE_DPI,
    LEGEND_BOX_ALPHA,
    LEGEND_BOX_EDGE,
    LEGEND_BOX_FACE,
    LIGHT_GREY,
    MID_GREY,
    THESIS_GRID_ALPHA,
    THESIS_LEGEND_SIZE,
    THESIS_PANEL_LABEL_SIZE,
    THESIS_TEXT_WIDTH_IN,
    TRAINING_TRACE_COLORS,
    dual_axis_bar_colors,
    inference_mode_color,
    style_legend_frame,
    thesis_panel_figure_size,
)
from pipeline2_analysis import (
    _add_uncertainty_note,
    _mean,
    _mode_cfg,
    _mode_order,
    _read_trace_csv,
    _safe_float,
    _safe_int,
    _show_milestone_error_bars,
    _show_trace_uncertainty_bands,
    _trace_band_alpha,
    _uncertainty_half_width,
)
from pipeline2_config import (
    aggregate_checkpoint_path,
    checkpoint_transfer_run_dir,
    get_inference_modes,
    get_seeds,
    run_dir,
    training_paths,
)
from pipeline_support import ensure_dir, load_json


def _mode_color_map(config: Dict[str, Any]) -> Dict[str, str]:
    """Return a stable thesis palette color per inference mode."""

    colors: Dict[str, str] = {}
    for idx, mode_name in enumerate(_mode_order(config)):
        mode = _mode_cfg(config, mode_name)
        colors[mode_name] = inference_mode_color(
            mode_name=mode_name,
            cutoff=mode.get("top_k_cutoff"),
            fallback_index=idx,
        )
    return colors


def _mode_compact_label(config: Dict[str, Any], mode_name: str) -> str:
    if str(mode_name) == "exact":
        return "Exact"
    mode = _mode_cfg(config, mode_name)
    cutoff = _safe_float(mode.get("top_k_cutoff"))
    if cutoff is not None:
        return f"Cutoff {cutoff:g}"
    return str(mode_name).replace("_", " ").capitalize()


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


def _draw_grouped_legend_box(
    legend_ax,
    legend_blocks: Sequence[Tuple[Sequence[Any], str | None, int]],
    *,
    x: float = 0.07,
    title_fontsize: float = THESIS_LEGEND_SIZE,
    entry_fontsize: float = THESIS_LEGEND_SIZE - 0.2,
) -> None:
    blocks = [(list(handles), title, max(1, int(ncol))) for handles, title, ncol in legend_blocks if list(handles)]
    if not blocks:
        legend_ax.axis("off")
        return
    legend_ax.axis("off")
    _legend_container_patch(legend_ax, x=x - 0.03, y=0.08, width=0.90, height=0.84)
    if len(blocks) == 1:
        title_positions, legend_positions = [0.86], [0.76]
    elif len(blocks) == 2:
        title_positions, legend_positions = [0.88, 0.50], [0.80, 0.42]
    else:
        title_positions, legend_positions = [0.90, 0.62, 0.33], [0.82, 0.54, 0.25]
    for idx, ((handles, title, ncol), title_y, legend_y) in enumerate(zip(blocks, title_positions, legend_positions)):
        if title:
            legend_ax.text(x, title_y, title, transform=legend_ax.transAxes, ha="left", va="top", fontsize=title_fontsize)
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
            columnspacing=1.0,
        )
        legend._legend_box.align = "left"
        if idx < len(blocks) - 1:
            legend_ax.add_artist(legend)


def _draw_horizontal_grouped_legend_box(
    fig,
    *,
    bounds: Tuple[float, float, float, float],
    legend_rows: Sequence[Tuple[Sequence[Any], str | None, int]],
    title_x: float = 0.04,
    legend_x: float = 0.28,
    title_fontsize: float = THESIS_LEGEND_SIZE,
    entry_fontsize: float = THESIS_LEGEND_SIZE - 0.15,
) -> None:
    rows = [(list(handles), title, max(1, int(ncol))) for handles, title, ncol in legend_rows if list(handles)]
    if not rows:
        return
    legend_ax = fig.add_axes(bounds)
    legend_ax.axis("off")
    _legend_container_patch(legend_ax, x=0.01, y=0.10, width=0.98, height=0.80)
    title_positions = [0.74, 0.34] if len(rows) == 2 else list(np.linspace(0.78, 0.24, len(rows)))
    legend_positions = [0.60, 0.20] if len(rows) == 2 else list(np.linspace(0.66, 0.12, len(rows)))
    for idx, ((handles, title, ncol), title_y, legend_y) in enumerate(zip(rows, title_positions, legend_positions)):
        if title:
            legend_ax.text(title_x, float(title_y), title, transform=legend_ax.transAxes, ha="left", va="top", fontsize=title_fontsize)
        legend = legend_ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(legend_x, float(legend_y)),
            ncol=ncol,
            fontsize=entry_fontsize,
            frameon=False,
            borderaxespad=0.0,
            labelspacing=0.48,
            handlelength=1.9,
            handletextpad=0.55,
            columnspacing=1.0,
        )
        legend._legend_box.align = "left"
        if idx < len(rows) - 1:
            legend_ax.add_artist(legend)


def _reached_milestone_rows(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    *,
    mode_name: str,
    milestone: float,
    required_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    """Return valid reached milestone rows, ordered by configured seed.

    Unreached or incomplete seeds are omitted.  This helper is used for
    censored milestone visualizations where a bar is still informative when
    only a subset of seeds reaches the requested milestone.
    """

    configured_seeds = get_seeds(config)
    configured_seed_set = set(configured_seeds)
    by_seed: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if str(row.get("mode_name")) != str(mode_name):
            continue
        row_milestone = _safe_float(row.get("milestone"))
        if row_milestone is None or float(row_milestone) != float(milestone):
            continue
        seed = _safe_int(row.get("seed"))
        if seed is None or seed not in configured_seed_set:
            continue
        if not bool(row.get("reached")):
            continue
        if any(row.get(field) in {None, ""} for field in required_fields):
            continue
        by_seed[seed] = row

    return [by_seed[seed] for seed in configured_seeds if seed in by_seed]


def _fully_reached_milestone_rows(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    *,
    mode_name: str,
    milestone: float,
    required_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    """Return one valid reached row per configured seed, or no rows."""

    reached_rows = _reached_milestone_rows(
        config,
        rows,
        mode_name=mode_name,
        milestone=milestone,
        required_fields=required_fields,
    )
    if len(reached_rows) != len(get_seeds(config)):
        return []
    return reached_rows


def _checkpoint_bar_axis_scaling(
    *,
    max_step_value: float,
    max_secondary_value: Optional[float] = None,
    exact_step_means: Sequence[float],
    exact_secondary_means: Optional[Sequence[float]] = None,
    max_time_value: Optional[float] = None,
    exact_time_means: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """Return (secondary_to_steps, left_axis_top) for the dual-axis checkpoint bar plot.

    The secondary bars are drawn on the left axis after conversion to step
    units. Therefore the left-axis limit must budget for both the outer step
    bars and the converted inner secondary-metric bars. Otherwise approximate
    runs can hit the plot ceiling even when the step bars fit.
    """

    if max_secondary_value is None:
        max_secondary_value = max_time_value
    if exact_secondary_means is None:
        exact_secondary_means = exact_time_means
    if max_secondary_value is None or exact_secondary_means is None:
        raise TypeError(
            "_checkpoint_bar_axis_scaling requires either secondary-metric arguments "
            "or the backward-compatible time arguments."
        )

    exact_pairs = [
        (float(step), float(value))
        for step, value in zip(exact_step_means, exact_secondary_means)
        if float(value) > 0.0
    ]
    if exact_pairs:
        numerator = sum(step * value for step, value in exact_pairs)
        denominator = sum(value * value for _, value in exact_pairs)
        exact_secondary_to_steps = (numerator / denominator) if denominator > 0.0 else None
    else:
        exact_secondary_to_steps = None

    if exact_secondary_to_steps and exact_secondary_to_steps > 0.0:
        secondary_to_steps = float(exact_secondary_to_steps)
    else:
        secondary_to_steps = float(max_step_value) / float(max_secondary_value)

    converted_secondary_max = float(max_secondary_value) * float(secondary_to_steps)
    content_top = max(float(max_step_value), converted_secondary_max)
    left_top = content_top * 1.10
    return secondary_to_steps, left_top


def _mix_with_white(color: Any, amount: float) -> Tuple[float, float, float]:
    amount = max(0.0, min(1.0, float(amount)))
    r, g, b = to_rgb(color)
    return (r + (1.0 - r) * amount, g + (1.0 - g) * amount, b + (1.0 - b) * amount)


def _darken_color(color: Any, amount: float) -> Tuple[float, float, float]:
    amount = max(0.0, min(1.0, float(amount)))
    r, g, b = to_rgb(color)
    factor = 1.0 - amount
    return (r * factor, g * factor, b * factor)


def _merge_series_dict(
    series_by_seed: Dict[int, Tuple[List[float], List[float]]],
    config: Dict[str, Any],
    *,
    clamp_unit_interval: bool = False,
) -> Tuple[List[float], List[float], List[float], List[float], List[int]]:
    merged: Dict[float, List[float]] = {}
    for xs, ys in series_by_seed.values():
        for x, y in zip(xs, ys):
            merged.setdefault(float(x), []).append(float(y))
    xs = sorted(merged)
    means: List[float] = []
    lowers: List[float] = []
    uppers: List[float] = []
    counts: List[int] = []
    for x in xs:
        values = merged[x]
        mean_value = _mean(values)
        if mean_value is None:
            continue
        half_width = _uncertainty_half_width(values, config)
        means.append(mean_value)
        counts.append(len(values))
        if half_width is None:
            lower = mean_value
            upper = mean_value
        else:
            lower = mean_value - half_width
            upper = mean_value + half_width
        lower = max(0.0, lower)
        if clamp_unit_interval:
            upper = min(1.0, upper)
        uppers.append(upper)
        lowers.append(lower)
    return xs, means, lowers, uppers, counts


def _require_checkpoint_transfer_v2_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    required_columns = {
        "target_step",
        "max_segment_cases_exact",
        "actual_end_step",
        "reached_target_checkpoint",
    }
    missing = [name for name in sorted(required_columns) if name not in rows[0]]
    if missing:
        raise RuntimeError(
            f"Stale checkpoint-transfer trace at {path}: missing {missing}. "
            "Rerun Pipeline II stage 'checkpoint-transfer' before visualizing."
        )

    early_stopped_segments: List[Tuple[int, int, int]] = []
    for row in rows:
        segment_index = _safe_int(row.get("segment_index"))
        target_step = _safe_int(row.get("target_step"))
        actual_end_step = _safe_int(row.get("actual_end_step"))
        if target_step is None or actual_end_step is None:
            continue
        if actual_end_step != target_step:
            early_stopped_segments.append((segment_index or -1, actual_end_step, target_step))
    if early_stopped_segments:
        preview = ", ".join(
            f"segment {segment}: {actual}/{target}"
            for segment, actual, target in early_stopped_segments[:4]
        )
        raise RuntimeError(
            f"Stale early-stopped checkpoint-transfer trace at {path} ({preview}). "
            "Checkpoint-transfer segments now run through the full exact anchor interval; "
            "rerun Pipeline II stage 'checkpoint-transfer' before visualizing."
        )


def _validated_checkpoint_transfer_rows(path: Path) -> List[Dict[str, Any]]:
    rows = _read_trace_csv(path)
    _require_checkpoint_transfer_v2_rows(path, rows)
    return rows


def _series_value_at_step(
    xs: Sequence[float],
    ys: Sequence[float],
    step: float,
) -> Optional[float]:
    """Return the displayed series value at ``step``.

    Training traces normally contain the exact integer checkpoint step. Linear
    interpolation is retained as a safe fallback for sparse traces, so marker
    placement still follows the visible curve rather than stale pre-smoothed
    checkpoint metadata.
    """

    points = sorted(
        (float(x), float(y))
        for x, y in zip(xs, ys)
    )
    if not points:
        return None

    target = float(step)
    for index, (x, y) in enumerate(points):
        if x == target:
            return y
        if x > target:
            if index == 0:
                return None
            left_x, left_y = points[index - 1]
            if x == left_x:
                return left_y
            weight = (target - left_x) / (x - left_x)
            return left_y + weight * (y - left_y)
    return None


def _project_checkpoint_values_onto_displayed_series(
    checkpoint_xs: Sequence[float],
    checkpoint_ys: Sequence[float],
    displayed_xs: Sequence[float],
    displayed_ys: Sequence[float],
) -> Tuple[List[float], List[float]]:
    """Place checkpoint markers on the currently displayed smoothed curve.

    Checkpoint steps are selected with ``checkpointing.rolling_window_updates``,
    while the figure can use a different
    ``visualization.trace_smoothing_window_points``. The stored checkpoint y
    value therefore cannot be used directly when those windows differ.
    """

    projected_xs: List[float] = []
    projected_ys: List[float] = []
    for raw_x, fallback_y in zip(checkpoint_xs, checkpoint_ys):
        x = float(raw_x)
        displayed_y = _series_value_at_step(displayed_xs, displayed_ys, x)
        projected_xs.append(x)
        projected_ys.append(float(fallback_y) if displayed_y is None else float(displayed_y))
    return projected_xs, projected_ys


def _pure_training_series_by_seed(
    config: Dict[str, Any],
    n_terms: int,
    mode_name: str,
    value_key: str,
    smooth_window: int,
) -> Dict[int, Tuple[List[float], List[float]]]:
    return _seed_trace_series(
        config,
        n_terms,
        mode_name,
        "train_trace.csv",
        value_key,
        smooth_window,
    )


def _checkpoint_transfer_segment_series_by_seed(
    config: Dict[str, Any],
    n_terms: int,
    mode_name: str,
    anchor_mode_name: str,
    value_key: str,
    smooth_window: int,
) -> List[Dict[int, Tuple[List[float], List[float]]]]:
    """Return green checkpoint-transfer series split by exact-checkpoint segment.

    Each segment is an independent approximate continuation from an exact
    checkpoint.  The visualization must therefore not smooth or draw lines across
    segment boundaries; otherwise the green curve looks like one continuous run
    and hides the restart points.
    """

    paths = training_paths(config)
    displayed_exact_by_seed = _pure_training_series_by_seed(
        config,
        n_terms,
        anchor_mode_name,
        value_key,
        smooth_window,
    )
    by_segment: Dict[int, Dict[int, Tuple[List[float], List[float]]]] = {}
    for seed in get_seeds(config):
        this_dir = checkpoint_transfer_run_dir(paths, seed, n_terms, mode_name, anchor_mode_name)
        segment_anchor_values: Dict[int, Tuple[float, float]] = {}
        segment_actual_end_step: Dict[int, int] = {}
        trace_path = this_dir / "checkpoint_transfer_trace.csv"
        for row in _validated_checkpoint_transfer_rows(trace_path):
            segment_index = _safe_int(row.get("segment_index"))
            anchor_step = _safe_float(row.get("anchor_step"))
            actual_end_step = _safe_int(row.get("actual_end_step"))
            if segment_index is None or anchor_step is None:
                continue
            if actual_end_step is not None:
                segment_actual_end_step[int(segment_index)] = int(actual_end_step)
            if value_key in {"true_mass", "true_mass_recent_mean", "zero_true_mass"}:
                anchor_value = _safe_float(row.get("anchor_rolling_true_mass_exact"))
            elif value_key == "loss":
                anchor_value = _safe_float(row.get("anchor_rolling_loss_exact"))
            else:
                anchor_value = None
            if anchor_value is not None:
                segment_anchor_values[int(segment_index)] = (float(anchor_step), float(anchor_value))

        train_rows = _read_trace_csv(this_dir / "checkpoint_transfer_train_trace.csv")
        per_segment_step: Dict[int, Dict[int, List[float]]] = {}
        for row in train_rows:
            segment_index = _safe_int(row.get("segment_index"))
            step = _safe_int(row.get("step"))
            value = _safe_float(row.get(value_key))
            if segment_index is None or step is None or value is None:
                continue
            max_step = segment_actual_end_step.get(int(segment_index))
            if max_step is not None and int(step) > int(max_step):
                continue
            per_segment_step.setdefault(int(segment_index), {}).setdefault(int(step), []).append(float(value))

        for segment_index, per_step in per_segment_step.items():
            xs: List[float] = []
            ys: List[float] = []
            for step in sorted(per_step):
                xs.append(float(step))
                ys.append(float(sum(per_step[step]) / len(per_step[step])))
            if not xs:
                continue
            # Smooth only within this segment, never across checkpoint boundaries.
            # Prefix points shorter than the full window are omitted, so every
            # post-anchor point is based on the same number of approximate
            # updates.  The exact anchor value is prepended separately so the
            # continuation visibly starts from the checkpoint it was initialized
            # from without pretending that the anchor is an approximate update.
            rolled_xs, rolled_ys = _full_window_rolling_series(xs, ys, smooth_window)
            anchor_point = segment_anchor_values.get(int(segment_index))
            if anchor_point is not None:
                anchor_x, anchor_y = anchor_point
                displayed_exact = displayed_exact_by_seed.get(int(seed))
                if displayed_exact is not None:
                    displayed_anchor_y = _series_value_at_step(
                        displayed_exact[0],
                        displayed_exact[1],
                        anchor_x,
                    )
                    if displayed_anchor_y is not None:
                        anchor_y = float(displayed_anchor_y)
                if not rolled_xs or float(anchor_x) < float(rolled_xs[0]):
                    rolled_xs = [anchor_x, *rolled_xs]
                    rolled_ys = [anchor_y, *rolled_ys]
            if not rolled_xs:
                continue
            by_segment.setdefault(segment_index, {})[int(seed)] = (rolled_xs, rolled_ys)

    return [by_segment[index] for index in sorted(by_segment)]


def _exact_posterior_checkpoint_markers(
    config: Dict[str, Any],
    n_terms: int,
    anchor_mode_name: str,
    value_key: str,
    *,
    displayed_xs: Sequence[float] = (),
    displayed_ys: Sequence[float] = (),
) -> Tuple[List[float], List[float]]:
    paths = training_paths(config)
    aggregate_path = aggregate_checkpoint_path(paths, n_terms, anchor_mode_name)
    if not aggregate_path.exists():
        return [], []
    payload = load_json(aggregate_path)
    metric_key = "rolling_true_mass" if value_key in {"true_mass", "true_mass_recent_mean"} else "rolling_loss"
    xs: List[float] = []
    ys: List[float] = []
    checkpoints = payload.get("posterior_checkpoints") or {}
    for threshold in sorted(checkpoints, key=lambda value: float(value)):
        info = checkpoints[threshold]
        if not bool(info.get("reached", False)):
            continue
        step = _safe_float(info.get("step"))
        value = _safe_float(info.get(metric_key))
        if step is None or value is None:
            continue
        xs.append(float(step))
        ys.append(float(value))
    if displayed_xs and displayed_ys:
        return _project_checkpoint_values_onto_displayed_series(
            xs,
            ys,
            displayed_xs,
            displayed_ys,
        )
    return xs, ys


def _plot_checkpoint_transfer_metric_on_axis(
    *,
    ax: Any,
    config: Dict[str, Any],
    n_terms: int,
    mode_name: str,
    anchor_mode_name: str,
    value_key: str,
    smooth_window: int,
    as_percent: bool = False,
    show_legend_label: bool = True,
) -> List[Any]:
    exact_xs, exact_means, exact_lowers, exact_uppers, _ = _merged_trace_stats(
        config,
        n_terms,
        anchor_mode_name,
        "train_trace.csv",
        value_key,
        smooth_window,
    )
    pure_xs, pure_means, pure_lowers, pure_uppers, _ = _merged_trace_stats(
        config,
        n_terms,
        mode_name,
        "train_trace.csv",
        value_key,
        smooth_window,
    )
    transfer_segments = _checkpoint_transfer_segment_series_by_seed(
        config,
        n_terms,
        mode_name,
        anchor_mode_name,
        value_key=value_key,
        smooth_window=smooth_window,
    )
    clamp = value_key in {"true_mass", "zero_true_mass"}
    colors = TRAINING_TRACE_COLORS
    draw_bands = _show_trace_uncertainty_bands(config, smooth_window)
    factor = 100.0 if as_percent else 1.0

    def scale_points(xs: List[float], ys: List[float], lowers: List[float], uppers: List[float]):
        return (
            [float(x) for x in xs],
            [factor * float(y) for y in ys],
            [factor * float(y) for y in lowers],
            [factor * float(y) for y in uppers],
        )

    x_exact, y_exact, l_exact, u_exact = scale_points(exact_xs, exact_means, exact_lowers, exact_uppers)
    x_pure, y_pure, l_pure, u_pure = scale_points(pure_xs, pure_means, pure_lowers, pure_uppers)

    exact_label = "Exact" if show_legend_label else None
    pure_label = f"{_mode_compact_label(config, mode_name)} from initialization" if show_legend_label else None
    ax.plot(x_exact, y_exact, color=colors["exact"], linewidth=1.45, label=exact_label, zorder=3)
    ax.plot(x_pure, y_pure, color=colors["pure"], linewidth=1.4, label=pure_label, zorder=3)

    checkpoint_x, checkpoint_y = _exact_posterior_checkpoint_markers(
        config,
        n_terms,
        anchor_mode_name,
        value_key,
        displayed_xs=exact_xs,
        displayed_ys=exact_means,
    )
    if checkpoint_x and checkpoint_y:
        x_marks = [float(x) for x in checkpoint_x]
        y_marks = [factor * float(y) for y in checkpoint_y]
        ax.scatter(
            x_marks,
            y_marks,
            marker="X",
            s=78,
            linewidths=0.9,
            edgecolors="white",
            color=colors["checkpoint"],
            label="Exact milestone anchors" if show_legend_label else None,
            zorder=6,
        )

    plotted_transfer_label = False
    for segment_series in transfer_segments:
        transfer_xs, transfer_means, transfer_lowers, transfer_uppers, _ = _merge_series_dict(
            segment_series,
            config,
            clamp_unit_interval=clamp,
        )
        if not transfer_xs:
            continue
        x_transfer, y_transfer, l_transfer, u_transfer = scale_points(
            transfer_xs,
            transfer_means,
            transfer_lowers,
            transfer_uppers,
        )
        ax.plot(
            x_transfer,
            y_transfer,
            color=colors["transfer"],
            linewidth=1.4,
            label=(
                f"{_mode_compact_label(config, mode_name)} checkpoint transfer"
                if show_legend_label and not plotted_transfer_label
                else None
            ),
            zorder=4,
        )
        if draw_bands and any(abs(u - l) > 0.0 for l, u in zip(l_transfer, u_transfer)):
            ax.fill_between(
                x_transfer,
                l_transfer,
                u_transfer,
                color=colors["transfer"],
                alpha=max(0.07, _trace_band_alpha(config) * 0.75),
                linewidth=0,
                zorder=2,
            )
        plotted_transfer_label = True

    if draw_bands:
        if any(abs(u - l) > 0.0 for l, u in zip(l_exact, u_exact)):
            ax.fill_between(x_exact, l_exact, u_exact, color=colors["exact"], alpha=_trace_band_alpha(config), linewidth=0, zorder=1)
        if any(abs(u - l) > 0.0 for l, u in zip(l_pure, u_pure)):
            ax.fill_between(x_pure, l_pure, u_pure, color=colors["pure"], alpha=_trace_band_alpha(config), linewidth=0, zorder=1)

    ax.grid(True, axis="y", alpha=THESIS_GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return [
        Line2D([0], [0], color=colors["exact"], linewidth=1.45, label="Exact"),
        Line2D([0], [0], color=colors["pure"], linewidth=1.4, label=f"{_mode_compact_label(config, mode_name)} from initialization"),
        Line2D([0], [0], color=colors["checkpoint"], marker="X", markersize=8, linewidth=0, label="Exact milestone anchors"),
        Line2D([0], [0], color=colors["transfer"], linewidth=1.4, label=f"{_mode_compact_label(config, mode_name)} checkpoint transfer"),
    ]


def _plot_combined_checkpoint_transfer_metric(
    *,
    config: Dict[str, Any],
    n_terms: int,
    mode_names: Sequence[str],
    anchor_mode_name: str,
    value_key: str,
    ylabel: str,
    output_path: Path,
    smooth_window: int,
    as_percent: bool = False,
    y_min_override: Optional[float] = None,
) -> None:
    if not mode_names:
        return

    ncols = 1 if len(mode_names) == 1 else 2
    nrows = (len(mode_names) + ncols - 1) // ncols
    fig, axes_grid = plt.subplots(
        nrows,
        ncols,
        figsize=thesis_panel_figure_size(nrows),
        squeeze=False,
        sharey=True,
    )
    axes = list(axes_grid.ravel())

    for idx, (ax, mode_name) in enumerate(zip(axes, mode_names)):
        _plot_checkpoint_transfer_metric_on_axis(
            ax=ax,
            config=config,
            n_terms=n_terms,
            mode_name=mode_name,
            anchor_mode_name=anchor_mode_name,
            value_key=value_key,
            smooth_window=smooth_window,
            as_percent=as_percent,
            show_legend_label=False,
        )
        ax.set_title(_mode_compact_label(config, mode_name), fontsize=THESIS_PANEL_LABEL_SIZE, pad=6)
        if value_key == "loss":
            ax.set_yscale("log")
        elif as_percent:
            ax.set_ylim(0.0, 100.0)
        elif value_key in {"true_mass", "true_mass_recent_mean"}:
            ax.set_ylim(0.0, 1.0)
        if idx % ncols != 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)

    if value_key == "loss":
        positive_y_values: List[float] = []
        for ax in axes:
            for line in ax.lines:
                for value in line.get_ydata(orig=False):
                    try:
                        y = float(value)
                    except (TypeError, ValueError):
                        continue
                    if y > 0.0:
                        positive_y_values.append(y)
            for collection in ax.collections:
                offsets = getattr(collection, "get_offsets", lambda: [])()
                for offset in offsets:
                    if len(offset) < 2:
                        continue
                    try:
                        y = float(offset[1])
                    except (TypeError, ValueError):
                        continue
                    if y > 0.0:
                        positive_y_values.append(y)
                for path in collection.get_paths():
                    for _x, raw_y in path.vertices:
                        try:
                            y = float(raw_y)
                        except (TypeError, ValueError):
                            continue
                        if y > 0.0:
                            positive_y_values.append(y)
        if positive_y_values:
            y_min = min(positive_y_values)
            y_max = max(positive_y_values)
            if y_min < y_max:
                # Multiplicative padding keeps the visual margin meaningful on
                # the logarithmic loss axis and prevents low-loss checkpoints
                # or continuation segments from being clipped.
                log_padding = (y_max / y_min) ** 0.05
                lower_bound = y_min / log_padding
                if y_min_override is not None:
                    lower_bound = max(lower_bound, y_min_override)
                for ax in axes:
                    ax.set_ylim(lower_bound, y_max * log_padding)

    colors = TRAINING_TRACE_COLORS
    legend_handles = [
        Line2D([0], [0], color=colors["exact"], linewidth=1.45, label="Exact"),
        Line2D([0], [0], color=colors["pure"], linewidth=1.4, label="Approximate from initialization"),
        Line2D(
            [0],
            [0],
            color=colors["checkpoint"],
            marker="X",
            markersize=8,
            linewidth=0,
            label="Exact milestone anchors",
        ),
        Line2D([0], [0], color=colors["transfer"], linewidth=1.4, label="Checkpoint transfer"),
    ]

    if len(mode_names) < len(axes):
        _draw_grouped_legend_box(
            axes[len(mode_names)],
            [(legend_handles, "Training trajectory", 1)],
            title_fontsize=THESIS_LEGEND_SIZE,
            entry_fontsize=THESIS_LEGEND_SIZE - 0.1,
        )
        for ax in axes[len(mode_names) + 1:]:
            ax.axis("off")
    else:
        _draw_horizontal_grouped_legend_box(
            fig,
            bounds=(0.12, 0.865, 0.76, 0.075),
            legend_rows=[(legend_handles, "Training trajectory", 2)],
            title_x=0.04,
            legend_x=0.32,
            title_fontsize=THESIS_LEGEND_SIZE - 0.05,
            entry_fontsize=THESIS_LEGEND_SIZE - 0.1,
        )
    fig.supxlabel("Training step", y=0.026, fontsize=THESIS_PANEL_LABEL_SIZE - 0.8)
    fig.supylabel(ylabel, x=0.032, fontsize=THESIS_PANEL_LABEL_SIZE - 0.8)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.13, hspace=0.34, wspace=0.16)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _mean_trajectory_milestone_intervals_from_rows(
    per_seed_rows: Dict[int, Sequence[Dict[str, Any]]],
    *,
    milestones: Sequence[float],
    rolling_window: int,
    secondary_key: str,
) -> Dict[float, Dict[str, Any]]:
    """Return milestone interval costs from the across-seed mean trajectory.

    This deliberately averages the complete per-step trajectories *before*
    finding milestone crossings.  It therefore describes the same mean training
    trajectory shown in the main true-sum-mass figure rather than averaging
    seed-specific first-crossing times.  The secondary quantity must be
    cumulative (elapsed time or model evaluations); its across-seed mean is read
    at the same aggregate crossing step and differenced between milestones.
    """

    if not per_seed_rows:
        return {}

    per_seed_by_step: Dict[int, Dict[int, Dict[str, float]]] = {}
    for seed, rows in per_seed_rows.items():
        by_step: Dict[int, Dict[str, float]] = {}
        for row in rows:
            step = _safe_int(row.get("step"))
            true_mass = _safe_float(row.get("true_mass"))
            secondary = _safe_float(row.get(secondary_key))
            if step is None or true_mass is None or secondary is None:
                continue
            by_step[int(step)] = {
                "true_mass": float(true_mass),
                "secondary": float(secondary),
            }
        if not by_step:
            return {}
        per_seed_by_step[int(seed)] = by_step

    common_steps = sorted(
        set.intersection(*(set(rows) for rows in per_seed_by_step.values()))
    )
    if not common_steps:
        return {}

    seed_count = len(per_seed_by_step)
    mean_true_mass = [
        sum(per_seed_by_step[seed][step]["true_mass"] for seed in per_seed_by_step) / seed_count
        for step in common_steps
    ]
    mean_secondary_by_step = {
        step: (
            sum(per_seed_by_step[seed][step]["secondary"] for seed in per_seed_by_step)
            / seed_count
        )
        for step in common_steps
    }
    rolling_steps, rolling_true_mass = _full_window_rolling_series(
        common_steps,
        mean_true_mass,
        max(1, int(rolling_window)),
    )

    crossings: Dict[float, int] = {}
    for milestone in sorted(float(value) for value in milestones):
        for step, posterior in zip(rolling_steps, rolling_true_mass):
            if float(posterior) >= float(milestone):
                crossings[float(milestone)] = int(step)
                break

    result: Dict[float, Dict[str, Any]] = {}
    previous_step = 0
    previous_secondary = 0.0
    previous_reached = True
    for milestone in sorted(float(value) for value in milestones):
        crossing_step = crossings.get(float(milestone))
        if crossing_step is None or not previous_reached:
            result[float(milestone)] = {
                "reached": False,
                "crossing_step": None,
                "step_delta": None,
                "secondary_delta": None,
                "seed_count": seed_count,
            }
            previous_reached = False
            continue

        current_secondary = float(mean_secondary_by_step[crossing_step])
        step_delta = float(crossing_step - previous_step)
        secondary_delta = float(current_secondary - previous_secondary)
        if step_delta < 0.0 or secondary_delta < 0.0:
            raise RuntimeError(
                "Mean-trajectory milestone measurements must be cumulative and nondecreasing: "
                f"milestone={milestone:g}, step_delta={step_delta:g}, "
                f"secondary_delta={secondary_delta:g}."
            )
        result[float(milestone)] = {
            "reached": True,
            "crossing_step": int(crossing_step),
            "step_delta": step_delta,
            "secondary_delta": secondary_delta,
            "seed_count": seed_count,
        }
        previous_step = int(crossing_step)
        previous_secondary = current_secondary

    return result


def _mean_trajectory_milestone_intervals(
    config: Dict[str, Any],
    *,
    n_terms: int,
    mode_name: str,
    milestones: Sequence[float],
    secondary_key: str,
) -> Dict[float, Dict[str, Any]]:
    """Load all configured pure-run traces and aggregate them before crossing."""

    paths = training_paths(config)
    per_seed_rows: Dict[int, Sequence[Dict[str, Any]]] = {}
    missing_seeds: List[int] = []
    for seed in get_seeds(config):
        trace_path = run_dir(paths, seed, n_terms, mode_name) / "train_trace.csv"
        trace_rows = _read_trace_csv(trace_path)
        if not trace_rows:
            missing_seeds.append(int(seed))
            continue
        per_seed_rows[int(seed)] = trace_rows

    if missing_seeds:
        raise RuntimeError(
            "Mean-trajectory milestone bars require the complete configured seed set. "
            f"Missing pure-run traces for mode={mode_name}, n_terms={n_terms}, seeds={missing_seeds}."
        )

    checkpoint_cfg = config.get("checkpointing") or {}
    rolling_window = int(checkpoint_cfg.get("rolling_window_updates", 100))
    return _mean_trajectory_milestone_intervals_from_rows(
        per_seed_rows,
        milestones=milestones,
        rolling_window=rolling_window,
        secondary_key=secondary_key,
    )



def _draw_checkpoint_cost_panel(
    *,
    ax: Any,
    config: Dict[str, Any],
    rows: List[Dict[str, Any]],
    n_terms: int,
    secondary_key: str,
    right_label: str,
    panel_label: str,
    show_x_ticklabels: bool,
) -> Tuple[List[Any], bool]:
    """Draw one milestone-cost panel while preserving the exact-bar alignment rule."""

    term_all_rows = [row for row in rows if int(row["n_terms"]) == int(n_terms)]
    if not term_all_rows:
        return [], False

    milestones = sorted({float(row["milestone"]) for row in term_all_rows})
    mode_names = _mode_order(config)
    x_positions = list(range(len(milestones)))
    mode_count = max(1, len(mode_names))
    bar_width = min(0.8 / mode_count, 0.16)
    group_width = bar_width * mode_count
    series: List[Dict[str, Any]] = []
    max_step_value = 0.0
    max_secondary_value = 0.0

    for mode_idx, mode_name in enumerate(mode_names):
        intervals = _mean_trajectory_milestone_intervals(
            config,
            n_terms=n_terms,
            mode_name=mode_name,
            milestones=milestones,
            secondary_key=secondary_key,
        )
        xs: List[float] = []
        step_means: List[float] = []
        secondary_means: List[float] = []
        reached_by_milestone: Dict[float, bool] = {}
        offset = -group_width / 2.0 + bar_width / 2.0 + mode_idx * bar_width
        for milestone_idx, milestone in enumerate(milestones):
            info = intervals.get(float(milestone), {})
            reached = bool(info.get("reached", False))
            reached_by_milestone[float(milestone)] = reached
            if not reached:
                continue
            step_delta = _safe_float(info.get("step_delta"))
            secondary_delta = _safe_float(info.get("secondary_delta"))
            if step_delta is None or secondary_delta is None:
                continue
            xs.append(float(x_positions[milestone_idx]) + offset)
            step_means.append(float(step_delta))
            secondary_means.append(float(secondary_delta))
            max_step_value = max(max_step_value, float(step_delta))
            max_secondary_value = max(max_secondary_value, float(secondary_delta))
        series.append(
            {
                "mode_name": mode_name,
                "offset": offset,
                "xs": xs,
                "step_means": step_means,
                "secondary_means": secondary_means,
                "reached_by_milestone": reached_by_milestone,
            }
        )

    plotted_series = [data for data in series if data["xs"]]
    if not plotted_series or max_step_value <= 0.0 or max_secondary_value <= 0.0:
        return [], False

    exact_series = next((data for data in plotted_series if str(data["mode_name"]) == "exact"), None)
    secondary_to_steps, left_top = _checkpoint_bar_axis_scaling(
        max_step_value=max_step_value,
        max_secondary_value=max_secondary_value,
        exact_step_means=exact_series["step_means"] if exact_series else [],
        exact_secondary_means=exact_series["secondary_means"] if exact_series else [],
    )
    ax_right = ax.secondary_yaxis(
        "right",
        functions=(
            lambda step_units: step_units / secondary_to_steps,
            lambda secondary: secondary * secondary_to_steps,
        ),
    )

    mode_handles: List[Any] = []
    has_unreached = False
    for data in series:
        mode_name = str(data["mode_name"])
        mode = _mode_cfg(config, mode_name)
        bar_colors = dual_axis_bar_colors(
            mode_name=mode_name,
            cutoff=mode.get("top_k_cutoff"),
            fallback_index=mode_names.index(mode_name) if mode_name in mode_names else 0,
        )
        base_color = bar_colors["base"]
        outer_color = bar_colors["outer"]
        inner_color = bar_colors["inner"]
        xs = data["xs"]
        step_means = data["step_means"]
        secondary_heights = [value * secondary_to_steps for value in data["secondary_means"]]
        if xs:
            ax.bar(
                xs,
                step_means,
                width=bar_width,
                color=outer_color,
                edgecolor=base_color,
                linewidth=1.0,
                zorder=2,
            )
            ax.bar(
                xs,
                secondary_heights,
                width=bar_width * 0.56,
                color=inner_color,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.97,
                zorder=3,
            )
        for milestone_idx, milestone in enumerate(milestones):
            if bool(data["reached_by_milestone"].get(float(milestone), False)):
                continue
            has_unreached = True
            x = float(x_positions[milestone_idx]) + float(data["offset"])
            ax.text(
                x,
                left_top * 0.012,
                "×",
                ha="center",
                va="bottom",
                fontsize=12.0,
                fontweight="semibold",
                color=base_color,
                zorder=5,
            )
        mode_handles.append(
            Patch(facecolor=inner_color, edgecolor="none", label=_mode_compact_label(config, mode_name))
        )

    ax.set_ylim(0.0, left_top)
    ax_right.set_ylabel(right_label)
    ax.set_title(panel_label, loc="left", fontsize=THESIS_PANEL_LABEL_SIZE, pad=6)
    ax.set_xticks(x_positions)
    interval_starts = [0.0, *milestones[:-1]]
    interval_labels = [
        (
            f"Init → {_format_milestone_label(end)}"
            if start == 0.0
            else f"{_format_milestone_label(start)} → {_format_milestone_label(end)}"
        )
        for start, end in zip(interval_starts, milestones)
    ]
    ax.set_xticklabels(interval_labels if show_x_ticklabels else ["" for _ in interval_labels])
    ax.grid(True, axis="y", alpha=THESIS_GRID_ALPHA, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)
    return mode_handles, has_unreached


def _plot_combined_dual_axis_checkpoint_costs(
    *,
    config: Dict[str, Any],
    rows: List[Dict[str, Any]],
    run_summaries: List[Dict[str, Any]],
    n_terms: int,
    output_path: Path,
) -> None:
    """Combine wall-clock and neural-evaluation milestone costs into one figure."""

    del run_summaries  # Inputs remain matched with the legacy plotting API.
    term_rows = [row for row in rows if int(row["n_terms"]) == int(n_terms)]
    if not term_rows:
        return

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(THESIS_TEXT_WIDTH_IN, 6.25),
        sharex=True,
        constrained_layout=False,
    )
    handles_top, unreached_top = _draw_checkpoint_cost_panel(
        ax=axes[0],
        config=config,
        rows=rows,
        n_terms=n_terms,
        secondary_key="elapsed_seconds",
        right_label="Wall-clock time (s)",
        panel_label="(a) Wall-clock time",
        show_x_ticklabels=False,
    )
    handles_bottom, unreached_bottom = _draw_checkpoint_cost_panel(
        ax=axes[1],
        config=config,
        rows=rows,
        n_terms=n_terms,
        secondary_key="read_mnist_model_evaluations_cumulative",
        right_label="Neural model evaluations",
        panel_label="(b) Neural model evaluations",
        show_x_ticklabels=True,
    )
    mode_handles = handles_top or handles_bottom
    if not mode_handles:
        plt.close(fig)
        return

    encoding_handles: List[Any] = [
        Patch(facecolor=LIGHT_GREY, edgecolor=MID_GREY, linewidth=1.0, label="Optimizer steps (outer)"),
        Patch(facecolor=MID_GREY, edgecolor="#44515F", linewidth=1.0, label="Panel cost (inner)"),
    ]
    if unreached_top or unreached_bottom:
        encoding_handles.append(
            Line2D([0], [0], color=MID_GREY, marker="x", markersize=7, linewidth=0, label="Not reached")
        )

    _draw_horizontal_grouped_legend_box(
        fig,
        bounds=(0.11, 0.865, 0.78, 0.09),
        legend_rows=[
            (mode_handles, "Inference mode", max(2, min(4, len(mode_handles)))),
            (encoding_handles, "Bar encoding", len(encoding_handles)),
        ],
        title_x=0.04,
        legend_x=0.28,
        title_fontsize=THESIS_LEGEND_SIZE - 0.05,
        entry_fontsize=THESIS_LEGEND_SIZE - 0.2,
    )
    fig.supxlabel("True-sum probability milestone interval", y=0.026, fontsize=THESIS_PANEL_LABEL_SIZE - 0.8)
    fig.supylabel("Optimizer steps", x=0.032, fontsize=THESIS_PANEL_LABEL_SIZE - 0.8)
    fig.subplots_adjust(left=0.13, right=0.87, top=0.82, bottom=0.11, hspace=0.34)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _plot_dual_axis_checkpoint_bars(
    *,
    config: Dict[str, Any],
    rows: List[Dict[str, Any]],
    run_summaries: List[Dict[str, Any]],
    n_terms: int,
    output_path: Path,
) -> None:
    term_all_rows = [row for row in rows if int(row["n_terms"]) == int(n_terms)]
    if not term_all_rows:
        return

    milestones = sorted({float(row["milestone"]) for row in term_all_rows})
    mode_names = _mode_order(config)
    x_positions = list(range(len(milestones)))
    mode_count = max(1, len(mode_names))
    bar_width = min(0.8 / mode_count, 0.16)
    group_width = bar_width * mode_count
    series: List[Dict[str, Any]] = []
    max_step_value = 0.0
    max_time_value = 0.0

    for mode_idx, mode_name in enumerate(mode_names):
        intervals = _mean_trajectory_milestone_intervals(
            config,
            n_terms=n_terms,
            mode_name=mode_name,
            milestones=milestones,
            secondary_key="elapsed_seconds",
        )
        xs: List[float] = []
        step_means: List[float] = []
        time_means: List[float] = []
        reached_by_milestone: Dict[float, bool] = {}
        offset = -group_width / 2.0 + bar_width / 2.0 + mode_idx * bar_width
        for milestone_idx, milestone in enumerate(milestones):
            info = intervals.get(float(milestone), {})
            reached = bool(info.get("reached", False))
            reached_by_milestone[float(milestone)] = reached
            if not reached:
                continue
            step_delta = _safe_float(info.get("step_delta"))
            time_delta = _safe_float(info.get("secondary_delta"))
            if step_delta is None or time_delta is None:
                continue
            xs.append(float(x_positions[milestone_idx]) + offset)
            step_means.append(float(step_delta))
            time_means.append(float(time_delta))
            max_step_value = max(max_step_value, float(step_delta))
            max_time_value = max(max_time_value, float(time_delta))

        series.append(
            {
                "mode_name": mode_name,
                "mode_idx": mode_idx,
                "offset": offset,
                "xs": xs,
                "step_means": step_means,
                "time_means": time_means,
                "reached_by_milestone": reached_by_milestone,
            }
        )

    plotted_series = [data for data in series if data["xs"]]
    if not plotted_series or max_step_value <= 0.0 or max_time_value <= 0.0:
        return

    exact_series = next((data for data in plotted_series if str(data["mode_name"]) == "exact"), None)
    time_to_steps, left_top = _checkpoint_bar_axis_scaling(
        max_step_value=max_step_value,
        max_secondary_value=max_time_value,
        exact_step_means=exact_series["step_means"] if exact_series else [],
        exact_secondary_means=exact_series["time_means"] if exact_series else [],
    )

    fig, ax = plt.subplots(figsize=(THESIS_TEXT_WIDTH_IN, 4.35))
    ax_right = ax.secondary_yaxis(
        "right",
        functions=(
            lambda step_units: step_units / time_to_steps,
            lambda seconds: seconds * time_to_steps,
        ),
    )

    mode_handles: List[Any] = []
    for data in series:
        mode_name = str(data["mode_name"])
        mode = _mode_cfg(config, mode_name)
        bar_colors = dual_axis_bar_colors(
            mode_name=mode_name,
            cutoff=mode.get("top_k_cutoff"),
            fallback_index=mode_names.index(mode_name) if mode_name in mode_names else 0,
        )
        base_color = bar_colors["base"]
        outer_color = bar_colors["outer"]
        inner_color = bar_colors["inner"]
        xs = data["xs"]
        step_means = data["step_means"]
        time_heights = [value * time_to_steps for value in data["time_means"]]

        if xs:
            ax.bar(
                xs,
                step_means,
                width=bar_width,
                color=outer_color,
                edgecolor=base_color,
                linewidth=1.0,
                zorder=2,
            )
            ax.bar(
                xs,
                time_heights,
                width=bar_width * 0.56,
                color=inner_color,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.97,
                zorder=3,
            )

        for milestone_idx, milestone in enumerate(milestones):
            if bool(data["reached_by_milestone"].get(float(milestone), False)):
                continue
            x = float(x_positions[milestone_idx]) + float(data["offset"])
            ax.text(
                x,
                left_top * 0.012,
                "×",
                ha="center",
                va="bottom",
                fontsize=13.0,
                fontweight="semibold",
                color=base_color,
                zorder=5,
            )

        mode_handles.append(
            Patch(facecolor=inner_color, edgecolor="none", label=_mode_compact_label(config, mode_name))
        )

    ax.set_ylim(0.0, left_top)
    ax.set_xlabel("True-sum probability milestone interval")
    ax.set_ylabel("Optimizer steps")
    ax_right.set_ylabel("Wall-clock time (s)")
    ax.set_xticks(x_positions)
    interval_starts = [0.0, *milestones[:-1]]
    ax.set_xticklabels(
        [
            (
                f"Init → {_format_milestone_label(end)}"
                if start == 0.0
                else f"{_format_milestone_label(start)} → {_format_milestone_label(end)}"
            )
            for start, end in zip(interval_starts, milestones)
        ]
    )
    ax.grid(True, axis="y", alpha=THESIS_GRID_ALPHA, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)

    step_metric_handle = Patch(
        facecolor=LIGHT_GREY,
        edgecolor=MID_GREY,
        linewidth=1.0,
        label="Optimizer steps (outer)",
    )
    time_metric_handle = Patch(
        facecolor=MID_GREY,
        edgecolor="#44515F",
        linewidth=1.0,
        label="Wall-clock time (inner)",
    )
    has_unreached = any(
        not bool(data["reached_by_milestone"].get(float(milestone), False))
        for data in series
        for milestone in milestones
    )
    metric_handles = [step_metric_handle, time_metric_handle]
    if has_unreached:
        metric_handles.append(
            Line2D(
                [0],
                [0],
                color=MID_GREY,
                marker="x",
                markersize=7,
                linewidth=0,
                label="Not reached",
            )
        )

    _draw_horizontal_grouped_legend_box(
        fig,
        bounds=(0.11, 0.865, 0.78, 0.09),
        legend_rows=[
            (mode_handles, "Inference mode", max(2, min(4, len(mode_handles)))),
            (metric_handles, "Bar encoding", len(metric_handles)),
        ],
        title_x=0.04,
        legend_x=0.28,
        title_fontsize=THESIS_LEGEND_SIZE - 0.05,
        entry_fontsize=THESIS_LEGEND_SIZE - 0.2,
    )
    fig.subplots_adjust(left=0.13, right=0.87, top=0.82, bottom=0.18)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _plot_dual_axis_checkpoint_model_evaluation_bars(
    *,
    config: Dict[str, Any],
    rows: List[Dict[str, Any]],
    run_summaries: List[Dict[str, Any]],
    n_terms: int,
    output_path: Path,
) -> None:
    term_all_rows = [row for row in rows if int(row["n_terms"]) == int(n_terms)]
    if not term_all_rows:
        return

    milestones = sorted({float(row["milestone"]) for row in term_all_rows})
    mode_names = _mode_order(config)
    x_positions = list(range(len(milestones)))
    mode_count = max(1, len(mode_names))
    bar_width = min(0.8 / mode_count, 0.16)
    group_width = bar_width * mode_count
    series: List[Dict[str, Any]] = []
    max_step_value = 0.0
    max_evaluation_value = 0.0

    for mode_idx, mode_name in enumerate(mode_names):
        intervals = _mean_trajectory_milestone_intervals(
            config,
            n_terms=n_terms,
            mode_name=mode_name,
            milestones=milestones,
            secondary_key="read_mnist_model_evaluations_cumulative",
        )
        xs: List[float] = []
        step_means: List[float] = []
        evaluation_means: List[float] = []
        reached_by_milestone: Dict[float, bool] = {}
        offset = -group_width / 2.0 + bar_width / 2.0 + mode_idx * bar_width
        for milestone_idx, milestone in enumerate(milestones):
            info = intervals.get(float(milestone), {})
            reached = bool(info.get("reached", False))
            reached_by_milestone[float(milestone)] = reached
            if not reached:
                continue
            step_delta = _safe_float(info.get("step_delta"))
            evaluation_delta = _safe_float(info.get("secondary_delta"))
            if step_delta is None or evaluation_delta is None:
                continue
            xs.append(float(x_positions[milestone_idx]) + offset)
            step_means.append(float(step_delta))
            evaluation_means.append(float(evaluation_delta))
            max_step_value = max(max_step_value, float(step_delta))
            max_evaluation_value = max(max_evaluation_value, float(evaluation_delta))

        series.append(
            {
                "mode_name": mode_name,
                "mode_idx": mode_idx,
                "offset": offset,
                "xs": xs,
                "step_means": step_means,
                "evaluation_means": evaluation_means,
                "reached_by_milestone": reached_by_milestone,
            }
        )

    plotted_series = [data for data in series if data["xs"]]
    if not plotted_series or max_step_value <= 0.0 or max_evaluation_value <= 0.0:
        return

    exact_series = next((data for data in plotted_series if str(data["mode_name"]) == "exact"), None)
    evaluations_to_steps, left_top = _checkpoint_bar_axis_scaling(
        max_step_value=max_step_value,
        max_secondary_value=max_evaluation_value,
        exact_step_means=exact_series["step_means"] if exact_series else [],
        exact_secondary_means=exact_series["evaluation_means"] if exact_series else [],
    )

    fig, ax = plt.subplots(figsize=(THESIS_TEXT_WIDTH_IN, 4.35))
    ax_right = ax.secondary_yaxis(
        "right",
        functions=(
            lambda step_units: step_units / evaluations_to_steps,
            lambda evaluations: evaluations * evaluations_to_steps,
        ),
    )

    mode_handles: List[Any] = []
    for data in series:
        mode_name = str(data["mode_name"])
        mode = _mode_cfg(config, mode_name)
        bar_colors = dual_axis_bar_colors(
            mode_name=mode_name,
            cutoff=mode.get("top_k_cutoff"),
            fallback_index=mode_names.index(mode_name) if mode_name in mode_names else 0,
        )
        base_color = bar_colors["base"]
        outer_color = bar_colors["outer"]
        inner_color = bar_colors["inner"]
        xs = data["xs"]
        step_means = data["step_means"]
        evaluation_heights = [value * evaluations_to_steps for value in data["evaluation_means"]]

        if xs:
            ax.bar(
                xs,
                step_means,
                width=bar_width,
                color=outer_color,
                edgecolor=base_color,
                linewidth=1.0,
                zorder=2,
            )
            ax.bar(
                xs,
                evaluation_heights,
                width=bar_width * 0.56,
                color=inner_color,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.97,
                zorder=3,
            )

        for milestone_idx, milestone in enumerate(milestones):
            if bool(data["reached_by_milestone"].get(float(milestone), False)):
                continue
            x = float(x_positions[milestone_idx]) + float(data["offset"])
            ax.text(
                x,
                left_top * 0.012,
                "×",
                ha="center",
                va="bottom",
                fontsize=13.0,
                fontweight="semibold",
                color=base_color,
                zorder=5,
            )

        mode_handles.append(
            Patch(facecolor=inner_color, edgecolor="none", label=_mode_compact_label(config, mode_name))
        )

    ax.set_ylim(0.0, left_top)
    ax.set_xlabel("True-sum probability milestone interval")
    ax.set_ylabel("Optimizer steps")
    ax_right.set_ylabel("Neural model evaluations")
    ax.set_xticks(x_positions)
    interval_starts = [0.0, *milestones[:-1]]
    ax.set_xticklabels(
        [
            (
                f"Init → {_format_milestone_label(end)}"
                if start == 0.0
                else f"{_format_milestone_label(start)} → {_format_milestone_label(end)}"
            )
            for start, end in zip(interval_starts, milestones)
        ]
    )
    ax.grid(True, axis="y", alpha=THESIS_GRID_ALPHA, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)

    step_metric_handle = Patch(
        facecolor=LIGHT_GREY,
        edgecolor=MID_GREY,
        linewidth=1.0,
        label="Optimizer steps (outer)",
    )
    evaluation_metric_handle = Patch(
        facecolor=MID_GREY,
        edgecolor="#44515F",
        linewidth=1.0,
        label="Neural model evaluations (inner)",
    )
    has_unreached = any(
        not bool(data["reached_by_milestone"].get(float(milestone), False))
        for data in series
        for milestone in milestones
    )
    metric_handles = [step_metric_handle, evaluation_metric_handle]
    if has_unreached:
        metric_handles.append(
            Line2D(
                [0],
                [0],
                color=MID_GREY,
                marker="x",
                markersize=7,
                linewidth=0,
                label="Not reached",
            )
        )

    _draw_horizontal_grouped_legend_box(
        fig,
        bounds=(0.11, 0.865, 0.78, 0.09),
        legend_rows=[
            (mode_handles, "Inference mode", max(2, min(4, len(mode_handles)))),
            (metric_handles, "Bar encoding", len(metric_handles)),
        ],
        title_x=0.04,
        legend_x=0.28,
        title_fontsize=THESIS_LEGEND_SIZE - 0.05,
        entry_fontsize=THESIS_LEGEND_SIZE - 0.2,
    )
    fig.subplots_adjust(left=0.13, right=0.87, top=0.82, bottom=0.18)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def _plot_checkpoint_transfer_metric_trajectory(
    *,
    config: Dict[str, Any],
    n_terms: int,
    mode_name: str,
    anchor_mode_name: str,
    value_key: str,
    ylabel: str,
    output_path: Path,
    smooth_window: int,
    as_percent: bool = False,
) -> None:
    exact_series = _pure_training_series_by_seed(config, n_terms, anchor_mode_name, value_key, smooth_window)
    pure_series = _pure_training_series_by_seed(config, n_terms, mode_name, value_key, smooth_window)
    transfer_segments = _checkpoint_transfer_segment_series_by_seed(
        config, n_terms, mode_name, anchor_mode_name, value_key, smooth_window
    )
    if not exact_series or not pure_series or not transfer_segments:
        return

    clamp = value_key in {"true_mass", "true_mass_recent_mean", "zero_true_mass"}
    exact_xs, exact_means, exact_lowers, exact_uppers, _ = _merge_series_dict(exact_series, config, clamp_unit_interval=clamp)
    pure_xs, pure_means, pure_lowers, pure_uppers, _ = _merge_series_dict(pure_series, config, clamp_unit_interval=clamp)
    if not exact_xs or not pure_xs:
        return

    fig, ax = plt.subplots(figsize=(THESIS_TEXT_WIDTH_IN, 3.75))
    colors = TRAINING_TRACE_COLORS
    draw_bands = _show_trace_uncertainty_bands(config, smooth_window)

    factor = 100.0 if as_percent else 1.0

    def scale_points(xs: List[float], ys: List[float], lowers: List[float], uppers: List[float]):
        return (
            [float(x) for x in xs],
            [factor * float(y) for y in ys],
            [factor * float(y) for y in lowers],
            [factor * float(y) for y in uppers],
        )

    x_exact, y_exact, l_exact, u_exact = scale_points(exact_xs, exact_means, exact_lowers, exact_uppers)
    x_pure, y_pure, l_pure, u_pure = scale_points(pure_xs, pure_means, pure_lowers, pure_uppers)

    ax.plot(x_exact, y_exact, color=colors["exact"], linewidth=2.0, label="Exact")
    ax.plot(
        x_pure,
        y_pure,
        color=colors["pure"],
        linewidth=2.0,
        label=f"{_mode_compact_label(config, mode_name)} from initialization",
    )

    checkpoint_x, checkpoint_y = _exact_posterior_checkpoint_markers(
        config,
        n_terms,
        anchor_mode_name,
        value_key,
        displayed_xs=exact_xs,
        displayed_ys=exact_means,
    )
    if checkpoint_x and checkpoint_y:
        x_marks = [float(x) for x in checkpoint_x]
        y_marks = [factor * float(y) for y in checkpoint_y]
        ax.scatter(
            x_marks,
            y_marks,
            marker="X",
            s=78,
            linewidths=0.9,
            edgecolors="white",
            color=colors["checkpoint"],
            label="Exact milestone anchors",
            zorder=6,
        )

    plotted_transfer_label = False
    for segment_series in transfer_segments:
        transfer_xs, transfer_means, transfer_lowers, transfer_uppers, _ = _merge_series_dict(
            segment_series,
            config,
            clamp_unit_interval=clamp,
        )
        if not transfer_xs:
            continue
        x_transfer, y_transfer, l_transfer, u_transfer = scale_points(
            transfer_xs,
            transfer_means,
            transfer_lowers,
            transfer_uppers,
        )
        ax.plot(
            x_transfer,
            y_transfer,
            color=colors["transfer"],
            linewidth=2.0,
            label=(
                f"{_mode_compact_label(config, mode_name)} checkpoint transfer"
                if not plotted_transfer_label
                else None
            ),
        )
        if draw_bands and any(abs(u - l) > 0.0 for l, u in zip(l_transfer, u_transfer)):
            ax.fill_between(
                x_transfer,
                l_transfer,
                u_transfer,
                color=colors["transfer"],
                alpha=max(0.08, _trace_band_alpha(config) * 0.8),
                linewidth=0,
            )
        plotted_transfer_label = True

    if draw_bands:
        if any(abs(u - l) > 0.0 for l, u in zip(l_exact, u_exact)):
            ax.fill_between(x_exact, l_exact, u_exact, color=colors["exact"], alpha=_trace_band_alpha(config), linewidth=0)
        if any(abs(u - l) > 0.0 for l, u in zip(l_pure, u_pure)):
            ax.fill_between(x_pure, l_pure, u_pure, color=colors["pure"], alpha=_trace_band_alpha(config), linewidth=0)

    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    if as_percent:
        ax.set_ylim(0.0, 105.0)
    elif value_key in {"true_mass", "true_mass_recent_mean"}:
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=THESIS_GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    style_legend_frame(ax.legend(loc="best", fontsize=THESIS_LEGEND_SIZE))
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _format_log_tick(value: float, _pos: int) -> str:
    if value <= 0:
        return ""
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:g}"
    if value >= 1:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2g}"


def _format_milestone_label(value: float) -> str:
    return f"{float(value):g}"


def _plot_metric_for_terms(
    *,
    config: Dict[str, Any],
    rows: List[Dict[str, Any]],
    run_summaries: List[Dict[str, Any]],
    n_terms: int,
    metric: str,
    ylabel: str,
    output_path: Path,
    maybe_log: bool = False,
) -> None:
    term_all_rows = [row for row in rows if int(row["n_terms"]) == int(n_terms)]
    if not term_all_rows:
        return

    milestones = sorted({float(row["milestone"]) for row in term_all_rows})
    mode_names = _mode_order(config)
    colors = _mode_color_map(config)
    x_positions = list(range(len(milestones)))
    mode_count = max(1, len(mode_names))
    bar_width = min(0.82 / mode_count, 0.18)
    group_width = bar_width * mode_count

    fig, ax = plt.subplots(figsize=(THESIS_TEXT_WIDTH_IN, 3.75))
    values_for_scale: List[float] = []
    plotted_modes: List[str] = []
    incomplete_markers: List[Tuple[float, str]] = []
    draw_error_bars = _show_milestone_error_bars(config)
    configured_seed_count = len(get_seeds(config))

    for mode_idx, mode_name in enumerate(mode_names):
        by_milestone: Dict[float, List[float]] = {}
        reached_count_by_milestone: Dict[float, int] = {}
        for milestone in milestones:
            reached_rows = _reached_milestone_rows(
                config,
                term_all_rows,
                mode_name=mode_name,
                milestone=milestone,
                required_fields=(metric,),
            )
            reached_count_by_milestone[float(milestone)] = len(reached_rows)
            if len(reached_rows) == configured_seed_count:
                by_milestone[milestone] = [float(row[metric]) for row in reached_rows]

        xs: List[float] = []
        ys: List[float] = []
        err_lowers: List[float] = []
        err_uppers: List[float] = []
        offset = -group_width / 2.0 + bar_width / 2.0 + mode_idx * bar_width
        for milestone_idx, milestone in enumerate(milestones):
            values = by_milestone.get(milestone)
            if not values:
                reached_count = reached_count_by_milestone.get(float(milestone), 0)
                incomplete_markers.append(
                    (float(x_positions[milestone_idx]) + offset, colors[mode_name])
                )
                continue
            mean_value = _mean(values)
            if mean_value is None:
                continue
            half_width = _uncertainty_half_width(values, config) if draw_error_bars else None
            xs.append(float(x_positions[milestone_idx]) + offset)
            ys.append(mean_value)
            if half_width is None:
                err_lowers.append(0.0)
                err_uppers.append(0.0)
            else:
                err_lowers.append(min(half_width, mean_value * 0.95) if maybe_log and mean_value > 0 else half_width)
                err_uppers.append(half_width)

        if not ys:
            continue
        for y_value, err_lower, err_upper in zip(ys, err_lowers, err_uppers):
            values_for_scale.append(y_value)
            if err_lower > 0:
                values_for_scale.append(max(0.0, y_value - err_lower))
            if err_upper > 0:
                values_for_scale.append(y_value + err_upper)
        plotted_modes.append(mode_name)
        yerr: Optional[List[List[float]]] = None
        if draw_error_bars and any(error > 0 for error in err_lowers + err_uppers):
            yerr = [err_lowers, err_uppers]
        ax.bar(
            xs,
            ys,
            width=bar_width,
            label=_mode_compact_label(config, mode_name),
            color=colors[mode_name],
            edgecolor="white",
            linewidth=0.8,
            yerr=yerr,
            error_kw={"elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0},
        )

    if maybe_log:
        positive = [v for v in values_for_scale if v > 0]
        if positive and max(positive) / min(positive) > 5.0:
            ax.set_yscale("log")
            ax.set_ylim(min(positive) * 0.75, max(positive) * 1.35)
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_log_tick))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("True-sum probability milestone")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_format_milestone_label(value) for value in milestones])
    ax.grid(True, axis="y", alpha=THESIS_GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for x, color in incomplete_markers:
        ax.text(
            x,
            0.025,
            "×",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10.0,
            color=color,
            fontweight="semibold",
            clip_on=False,
        )

    if plotted_modes:
        handles, labels = ax.get_legend_handles_labels()
        if incomplete_markers:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=MID_GREY,
                    marker="x",
                    markersize=6.5,
                    linewidth=0,
                    label="Not reached by all seeds",
                )
            )
            labels.append("Not reached by all seeds")
        style_legend_frame(ax.legend(handles, labels, loc="best", fontsize=THESIS_LEGEND_SIZE))

    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _full_window_rolling_series(
    xs: Sequence[float],
    ys: Sequence[float],
    window: int,
) -> Tuple[List[float], List[float]]:
    """Return a trailing rolling series with exactly ``window`` observations.

    Prefix points with fewer than ``window`` observations are omitted.  This
    means every displayed smoothed point and checkpoint crossing is based on the
    same amount of evidence instead of using short unstable prefix averages.
    """

    max_window = max(1, int(window))
    if max_window <= 1:
        return [float(x) for x in xs], [float(y) for y in ys]

    rolled_xs: List[float] = []
    rolled_ys: List[float] = []
    running: Deque[float] = deque()
    total = 0.0
    for x, raw_y in zip(xs, ys):
        y = float(raw_y)
        running.append(y)
        total += y
        while len(running) > max_window:
            total -= running.popleft()
        if len(running) == max_window:
            rolled_xs.append(float(x))
            rolled_ys.append(float(total / max_window))
    return rolled_xs, rolled_ys


def _seed_trace_series(
    config: Dict[str, Any],
    n_terms: int,
    mode_name: str,
    trace_name: str,
    value_key: str,
    smooth_window: int,
) -> Dict[int, Tuple[List[int], List[float]]]:
    paths = training_paths(config)
    by_seed: Dict[int, Tuple[List[int], List[float]]] = {}
    for seed in get_seeds(config):
        per_step: Dict[int, List[float]] = {}
        rows = _read_trace_csv(run_dir(paths, seed, n_terms, mode_name) / trace_name)
        for row in rows:
            value = _safe_float(row.get(value_key))
            step = _safe_int(row.get("step"))
            if value is None or step is None:
                continue
            per_step.setdefault(step, []).append(value)
        if not per_step:
            continue
        xs = [float(x) for x in sorted(per_step)]
        ys = [float(sum(per_step[int(x)]) / len(per_step[int(x)])) for x in xs]
        rolled_xs, rolled_ys = _full_window_rolling_series(xs, ys, smooth_window)
        if not rolled_xs:
            continue
        by_seed[int(seed)] = (rolled_xs, rolled_ys)
    return by_seed


def _merged_trace_stats(
    config: Dict[str, Any],
    n_terms: int,
    mode_name: str,
    trace_name: str,
    value_key: str,
    smooth_window: int,
) -> Tuple[List[int], List[float], List[float], List[float], List[int]]:
    merged: Dict[int, List[float]] = {}
    for xs, ys in _seed_trace_series(config, n_terms, mode_name, trace_name, value_key, smooth_window).values():
        for step, value in zip(xs, ys):
            merged.setdefault(step, []).append(float(value))

    xs = sorted(merged)
    means: List[float] = []
    lowers: List[float] = []
    uppers: List[float] = []
    counts: List[int] = []
    for step in xs:
        values = merged[step]
        mean_value = _mean(values)
        if mean_value is None:
            continue
        half_width = _uncertainty_half_width(values, config)
        means.append(mean_value)
        counts.append(len(values))
        if half_width is None:
            lowers.append(mean_value)
            uppers.append(mean_value)
        else:
            lower = mean_value - half_width
            upper = mean_value + half_width
            if value_key in {"loss", "true_mass", "branch_count", "zero_true_mass"}:
                lower = max(0.0, lower)
            if value_key in {"true_mass", "zero_true_mass"}:
                upper = min(1.0, upper)
            lowers.append(lower)
            uppers.append(upper)
    return xs, means, lowers, uppers, counts



def _combine_vertical_figure_assets(
    *,
    input_paths: Sequence[Path],
    output_path: Path,
    panel_labels: Sequence[str],
    height_ratios: Sequence[float],
    figure_height: float = 7.15,
) -> None:
    """Stack already generated thesis plots into one reproducible appendix plate."""

    if len(input_paths) != len(panel_labels) or len(input_paths) != len(height_ratios):
        raise ValueError("input_paths, panel_labels, and height_ratios must have matching lengths")
    if not input_paths or any(not path.exists() for path in input_paths):
        return
    fig = plt.figure(figsize=(THESIS_TEXT_WIDTH_IN, float(figure_height)), constrained_layout=False)
    grid = fig.add_gridspec(len(input_paths), 1, height_ratios=list(height_ratios), hspace=0.10)
    for idx, (path, label) in enumerate(zip(input_paths, panel_labels)):
        ax = fig.add_subplot(grid[idx, 0])
        ax.imshow(plt.imread(path))
        ax.set_axis_off()
        ax.text(
            0.002,
            1.002,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=THESIS_PANEL_LABEL_SIZE,
        )
    ensure_dir(output_path.parent)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.985, bottom=0.005)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _plot_trace(
    *,
    config: Dict[str, Any],
    n_terms: int,
    trace_name: str,
    value_key: str,
    ylabel: str,
    output_path: Path,
    smooth_window: int = 1,
    legend_loc: str = "upper left",
    legend_bbox: Optional[Tuple[float, float]] = (1.02, 1.0),
    show_footer: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(THESIS_TEXT_WIDTH_IN, 3.55))
    plotted = False
    colors = _mode_color_map(config)
    draw_bands = _show_trace_uncertainty_bands(config, smooth_window)
    for mode_name in _mode_order(config):
        xs, ys, lowers, uppers, _counts = _merged_trace_stats(
            config,
            n_terms,
            mode_name,
            trace_name,
            value_key,
            smooth_window,
        )
        if not xs:
            continue
        if draw_bands and any(abs(upper - lower) > 0.0 for lower, upper in zip(lowers, uppers)):
            ax.fill_between(
                xs,
                lowers,
                uppers,
                color=colors[mode_name],
                alpha=max(0.08, _trace_band_alpha(config) * 0.72),
                linewidth=0,
                zorder=1,
            )
        ax.plot(
            xs,
            ys,
            label=_mode_compact_label(config, mode_name),
            linewidth=2.15,
            color=colors[mode_name],
            zorder=3,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=THESIS_GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_kwargs: Dict[str, Any] = {"loc": legend_loc, "fontsize": THESIS_LEGEND_SIZE}
    if legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox
    style_legend_frame(ax.legend(**legend_kwargs))
    if show_footer:
        _add_uncertainty_note(fig, config, enabled=draw_bands)
        if draw_bands:
            fig.tight_layout(rect=(0, 0.045, 1, 1))
        else:
            fig.tight_layout()
    else:
        fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)






if __name__ == "__main__":
    main()
