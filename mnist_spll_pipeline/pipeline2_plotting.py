from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_palette import (
    A4_PAGE_WIDTH_IN,
    FIGURE_DPI,
    LIGHT_GREY,
    MID_GREY,
    TRAINING_TRACE_COLORS,
    dual_axis_bar_colors,
    inference_mode_color,
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
        return "exact"
    mode = _mode_cfg(config, mode_name)
    cutoff = _safe_float(mode.get("top_k_cutoff"))
    if cutoff is not None:
        return f"cutoff {cutoff:g}"
    if bool(mode.get("adaptive_top_k", False)):
        target = _safe_float(mode.get("posterior_mass_target"))
        if target is not None:
            return f"mass {target:g}"
    return str(mode_name).replace("_", " ")


def _fully_reached_milestone_rows(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    *,
    mode_name: str,
    milestone: float,
    required_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    """Return one valid reached row per configured seed, or no rows.

    A milestone bar must never summarize only the successful subset of seeds.
    Returning an empty list when any configured seed is missing, unreached, or
    lacks a required metric makes incomplete mode/checkpoint pairs disappear
    from both the main dual-axis figure and the appendix bar figures.
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
        by_seed[seed] = row

    if set(by_seed) != configured_seed_set:
        return []

    ordered_rows = [by_seed[seed] for seed in configured_seeds]
    if any(not bool(row.get("reached")) for row in ordered_rows):
        return []
    if any(row.get(field) in {None, ""} for row in ordered_rows for field in required_fields):
        return []
    return ordered_rows


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

    exact_label = f"Pure {anchor_mode_name}" if show_legend_label else None
    pure_label = f"Pure {mode_name}" if show_legend_label else None
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
            label="Aggregate exact checkpoints" if show_legend_label else None,
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
                f"{mode_name} from {anchor_mode_name} checkpoints"
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

    ax.grid(True, axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return [
        Line2D([0], [0], color=colors["exact"], linewidth=1.45, label=f"Pure {anchor_mode_name}"),
        Line2D([0], [0], color=colors["pure"], linewidth=1.4, label=f"Pure {mode_name}"),
        Line2D([0], [0], color=colors["checkpoint"], marker="X", markersize=8, linewidth=0, label="Aggregate exact checkpoints"),
        Line2D([0], [0], color=colors["transfer"], linewidth=1.4, label=f"{mode_name} from {anchor_mode_name} checkpoints"),
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
) -> None:
    if not mode_names:
        return

    fig, axes = plt.subplots(
        1,
        len(mode_names),
        figsize=(max(A4_PAGE_WIDTH_IN, 4.1 * len(mode_names) + 0.9), 4.35),
        sharey=True,
    )
    if not isinstance(axes, (list, tuple)):
        import numpy as _np  # type: ignore
        if isinstance(axes, _np.ndarray):
            axes = list(axes.ravel())
        else:
            axes = [axes]

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
        ax.set_title(_mode_compact_label(config, mode_name), fontsize=11, pad=8)
        ax.set_xlabel("Training iteration")
        if value_key == "loss":
            ax.set_yscale("log")
        elif as_percent:
            ax.set_ylim(0.0, 100.0)
        if idx != 0:
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
                for ax in axes:
                    ax.set_ylim(y_min / log_padding, y_max * log_padding)

    colors = TRAINING_TRACE_COLORS
    legend_handles = [
        Line2D([0], [0], color=colors["exact"], linewidth=1.05, label="Pure exact"),
        Line2D([0], [0], color=colors["pure"], linewidth=1.0, label="Pure approximate"),
        Line2D(
            [0],
            [0],
            color=colors["checkpoint"],
            marker="X",
            markersize=8,
            linewidth=0,
            label="Exact checkpoints",
        ),
        Line2D([0], [0], color=colors["transfer"], linewidth=1.0, label="Approx. continuation"),
    ]

    title = f"{n_terms}-term SPLL training — {ylabel}"
    fig.text(0.5, 0.978, title, ha="center", va="top", fontsize=15)
    fig.text(0.015, 0.48, ylabel, ha="center", va="center", rotation="vertical", fontsize=11)
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
        fontsize=9,
        columnspacing=1.15,
        handletextpad=0.55,
    )
    legend.set_in_layout(False)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.79, bottom=0.14, wspace=0.08)
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
    colors = _mode_color_map(config)
    x_positions = list(range(len(milestones)))
    mode_count = max(1, len(mode_names))
    bar_width = min(0.8 / mode_count, 0.16)
    group_width = bar_width * mode_count
    series: List[Dict[str, Any]] = []
    max_step_value = 0.0
    max_time_value = 0.0
    for mode_idx, mode_name in enumerate(mode_names):
        by_milestone_steps: Dict[float, List[float]] = {}
        by_milestone_time: Dict[float, List[float]] = {}
        for milestone_idx, milestone in enumerate(milestones):
            current_rows = _fully_reached_milestone_rows(
                config,
                term_all_rows,
                mode_name=mode_name,
                milestone=milestone,
                required_fields=("step", "elapsed_seconds"),
            )
            if not current_rows:
                continue

            current_by_seed = {
                int(row["seed"]): row
                for row in current_rows
            }
            if milestone_idx == 0:
                previous_by_seed: Dict[int, Dict[str, Any]] = {}
            else:
                previous_milestone = milestones[milestone_idx - 1]
                previous_rows = _fully_reached_milestone_rows(
                    config,
                    term_all_rows,
                    mode_name=mode_name,
                    milestone=previous_milestone,
                    required_fields=("step", "elapsed_seconds"),
                )
                if not previous_rows:
                    continue
                previous_by_seed = {
                    int(row["seed"]): row
                    for row in previous_rows
                }

            step_values: List[float] = []
            time_values: List[float] = []
            for seed in get_seeds(config):
                current_row = current_by_seed[int(seed)]
                previous_row = previous_by_seed.get(int(seed))
                previous_step = float(previous_row["step"]) if previous_row is not None else 0.0
                previous_time = float(previous_row["elapsed_seconds"]) if previous_row is not None else 0.0
                step_delta = float(current_row["step"]) - previous_step
                time_delta = float(current_row["elapsed_seconds"]) - previous_time
                if step_delta < 0.0 or time_delta < 0.0:
                    previous_label = 0.0 if milestone_idx == 0 else milestones[milestone_idx - 1]
                    raise RuntimeError(
                        "Checkpoint measurements must be cumulative and nondecreasing before "
                        "interval conversion: "
                        f"mode={mode_name}, seed={seed}, interval={previous_label:g}-{milestone:g}, "
                        f"step_delta={step_delta:g}, time_delta={time_delta:g}."
                    )
                step_values.append(step_delta)
                time_values.append(time_delta)

            by_milestone_steps[milestone] = step_values
            by_milestone_time[milestone] = time_values

        xs: List[float] = []
        step_means: List[float] = []
        time_means: List[float] = []
        offset = -group_width / 2.0 + bar_width / 2.0 + mode_idx * bar_width
        for milestone_idx, milestone in enumerate(milestones):
            step_values = by_milestone_steps.get(milestone)
            time_values = by_milestone_time.get(milestone)
            if not step_values or not time_values:
                continue
            step_mean = _mean(step_values)
            time_mean = _mean(time_values)
            if step_mean is None or time_mean is None:
                continue
            xs.append(float(x_positions[milestone_idx]) + offset)
            step_means.append(float(step_mean))
            time_means.append(float(time_mean))
            max_step_value = max(max_step_value, float(step_mean))
            max_time_value = max(max_time_value, float(time_mean))

        if xs:
            series.append(
                {
                    "mode_name": mode_name,
                    "xs": xs,
                    "step_means": step_means,
                    "time_means": time_means,
                }
            )

    if not series or max_step_value <= 0.0 or max_time_value <= 0.0:
        return

    exact_series = next((data for data in series if str(data["mode_name"]) == "exact"), None)
    time_to_steps, left_top = _checkpoint_bar_axis_scaling(
        max_step_value=max_step_value,
        max_secondary_value=max_time_value,
        exact_step_means=exact_series["step_means"] if exact_series else [],
        exact_secondary_means=exact_series["time_means"] if exact_series else [],
    )

    fig, ax = plt.subplots(figsize=(max(A4_PAGE_WIDTH_IN, 9.2, 0.9 * len(milestones) + 3.4), 6.05))
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
        mode_handles.append(
            Patch(facecolor=inner_color, edgecolor="none", label=_mode_compact_label(config, mode_name))
        )

    ax.set_ylim(0.0, left_top)
    ax.set_xlabel("Training true-sum posterior interval")
    ax.set_ylabel("Steps in interval")
    ax_right.set_ylabel("Wall-clock time in interval (s)")
    ax.set_xticks(x_positions)
    interval_starts = [0.0, *milestones[:-1]]
    ax.set_xticklabels(
        [
            f"{_format_milestone_label(start)}–{_format_milestone_label(end)}"
            for start, end in zip(interval_starts, milestones)
        ]
    )
    ax.grid(True, axis="y", alpha=0.18, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)

    step_metric_handle = Patch(
        facecolor=LIGHT_GREY,
        edgecolor=MID_GREY,
        linewidth=1.0,
        label="Steps: outer bar (left axis)",
    )
    time_metric_handle = Patch(
        facecolor=MID_GREY,
        edgecolor="#44515F",
        linewidth=1.0,
        label="Wall-clock: inner bar (right axis)",
    )

    fig.text(0.5, 0.978, f"{n_terms}-term SPLL training — Posterior checkpoint intervals", ha="center", va="top", fontsize=15)
    mode_legend = fig.legend(
        handles=mode_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=max(2, min(4, len(mode_handles))),
        frameon=False,
        fontsize=9,
        columnspacing=1.25,
        handletextpad=0.55,
    )
    mode_legend.set_in_layout(False)
    metric_legend = fig.legend(
        handles=[step_metric_handle, time_metric_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=2,
        frameon=False,
        fontsize=8.7,
        columnspacing=1.8,
        handletextpad=0.7,
    )
    metric_legend.set_in_layout(False)

    fig.subplots_adjust(left=0.09, right=0.91, top=0.80, bottom=0.14)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _plot_dual_axis_checkpoint_lookup_bars(
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
    max_lookup_value = 0.0
    for mode_idx, mode_name in enumerate(mode_names):
        by_milestone_steps: Dict[float, List[float]] = {}
        by_milestone_lookups: Dict[float, List[float]] = {}
        for milestone_idx, milestone in enumerate(milestones):
            current_rows = _fully_reached_milestone_rows(
                config,
                term_all_rows,
                mode_name=mode_name,
                milestone=milestone,
                required_fields=("step", "cumulative_read_mnist_lookup_calls"),
            )
            if not current_rows:
                continue

            current_by_seed = {int(row["seed"]): row for row in current_rows}
            if milestone_idx == 0:
                previous_by_seed: Dict[int, Dict[str, Any]] = {}
            else:
                previous_milestone = milestones[milestone_idx - 1]
                previous_rows = _fully_reached_milestone_rows(
                    config,
                    term_all_rows,
                    mode_name=mode_name,
                    milestone=previous_milestone,
                    required_fields=("step", "cumulative_read_mnist_lookup_calls"),
                )
                if not previous_rows:
                    continue
                previous_by_seed = {int(row["seed"]): row for row in previous_rows}

            step_values: List[float] = []
            lookup_values: List[float] = []
            for seed in get_seeds(config):
                current_row = current_by_seed[int(seed)]
                previous_row = previous_by_seed.get(int(seed))
                previous_step = float(previous_row["step"]) if previous_row is not None else 0.0
                previous_lookup = (
                    float(previous_row["cumulative_read_mnist_lookup_calls"])
                    if previous_row is not None
                    else 0.0
                )
                step_delta = float(current_row["step"]) - previous_step
                lookup_delta = float(current_row["cumulative_read_mnist_lookup_calls"]) - previous_lookup
                if step_delta < 0.0 or lookup_delta < 0.0:
                    previous_label = 0.0 if milestone_idx == 0 else milestones[milestone_idx - 1]
                    raise RuntimeError(
                        "Checkpoint measurements must be cumulative and nondecreasing before "
                        "interval conversion: "
                        f"mode={mode_name}, seed={seed}, interval={previous_label:g}-{milestone:g}, "
                        f"step_delta={step_delta:g}, lookup_delta={lookup_delta:g}."
                    )
                step_values.append(step_delta)
                lookup_values.append(lookup_delta)

            by_milestone_steps[milestone] = step_values
            by_milestone_lookups[milestone] = lookup_values

        xs: List[float] = []
        step_means: List[float] = []
        lookup_means: List[float] = []
        offset = -group_width / 2.0 + bar_width / 2.0 + mode_idx * bar_width
        for milestone_idx, milestone in enumerate(milestones):
            step_values = by_milestone_steps.get(milestone)
            lookup_values = by_milestone_lookups.get(milestone)
            if not step_values or not lookup_values:
                continue
            step_mean = _mean(step_values)
            lookup_mean = _mean(lookup_values)
            if step_mean is None or lookup_mean is None:
                continue
            xs.append(float(x_positions[milestone_idx]) + offset)
            step_means.append(float(step_mean))
            lookup_means.append(float(lookup_mean))
            max_step_value = max(max_step_value, float(step_mean))
            max_lookup_value = max(max_lookup_value, float(lookup_mean))

        if xs:
            series.append(
                {
                    "mode_name": mode_name,
                    "xs": xs,
                    "step_means": step_means,
                    "lookup_means": lookup_means,
                }
            )

    if not series or max_step_value <= 0.0 or max_lookup_value <= 0.0:
        return

    exact_series = next((data for data in series if str(data["mode_name"]) == "exact"), None)
    lookup_to_steps, left_top = _checkpoint_bar_axis_scaling(
        max_step_value=max_step_value,
        max_secondary_value=max_lookup_value,
        exact_step_means=exact_series["step_means"] if exact_series else [],
        exact_secondary_means=exact_series["lookup_means"] if exact_series else [],
    )

    fig, ax = plt.subplots(figsize=(max(A4_PAGE_WIDTH_IN, 9.2, 0.9 * len(milestones) + 3.4), 6.05))
    ax_right = ax.secondary_yaxis(
        "right",
        functions=(
            lambda step_units: step_units / lookup_to_steps,
            lambda lookups: lookups * lookup_to_steps,
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
        lookup_heights = [value * lookup_to_steps for value in data["lookup_means"]]

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
            lookup_heights,
            width=bar_width * 0.56,
            color=inner_color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.97,
            zorder=3,
        )
        mode_handles.append(
            Patch(facecolor=inner_color, edgecolor="none", label=_mode_compact_label(config, mode_name))
        )

    ax.set_ylim(0.0, left_top)
    ax.set_xlabel("Training true-sum posterior interval")
    ax.set_ylabel("Steps in interval")
    ax_right.set_ylabel("Generated readMNist calls in interval")
    ax.set_xticks(x_positions)
    interval_starts = [0.0, *milestones[:-1]]
    ax.set_xticklabels(
        [
            f"{_format_milestone_label(start)}–{_format_milestone_label(end)}"
            for start, end in zip(interval_starts, milestones)
        ]
    )
    ax.grid(True, axis="y", alpha=0.18, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)

    step_metric_handle = Patch(
        facecolor=LIGHT_GREY,
        edgecolor=MID_GREY,
        linewidth=1.0,
        label="Steps: outer bar (left axis)",
    )
    lookup_metric_handle = Patch(
        facecolor=MID_GREY,
        edgecolor="#44515F",
        linewidth=1.0,
        label="Lookup count: inner bar (right axis)",
    )

    fig.text(0.5, 0.978, f"{n_terms}-term SPLL training — Posterior checkpoint intervals (lookups)", ha="center", va="top", fontsize=15)
    mode_legend = fig.legend(
        handles=mode_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=max(2, min(4, len(mode_handles))),
        frameon=False,
        fontsize=9,
        columnspacing=1.25,
        handletextpad=0.55,
    )
    mode_legend.set_in_layout(False)
    metric_legend = fig.legend(
        handles=[step_metric_handle, lookup_metric_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=2,
        frameon=False,
        fontsize=8.7,
        columnspacing=1.8,
        handletextpad=0.7,
    )
    metric_legend.set_in_layout(False)

    fig.subplots_adjust(left=0.09, right=0.91, top=0.80, bottom=0.14)
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

    fig, ax = plt.subplots(figsize=(max(A4_PAGE_WIDTH_IN, 8.8), 5.2))
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

    ax.plot(x_exact, y_exact, color=colors["exact"], linewidth=2.0, label=f"Pure {anchor_mode_name}")
    ax.plot(x_pure, y_pure, color=colors["pure"], linewidth=2.0, label=f"Pure {mode_name}")

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
            label="Aggregate exact checkpoints",
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
                f"{mode_name} from {anchor_mode_name} checkpoints"
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

    ax.set_title(f"{n_terms}-term SPLL training: {ylabel}", loc="left")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(ylabel)
    if as_percent:
        ax.set_ylim(0.0, 105.0)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False)
    fig.text(
        0.01,
        0.015,
        "Purple X markers are aggregate exact checkpoints. Green pieces start at exact anchors, then show full-window approximate continuations until the next posterior target is reached or the exact segment budget is exhausted.",
        ha="left",
        va="bottom",
        fontsize=8,
    )
    _add_uncertainty_note(fig, config, enabled=draw_bands, y=0.034)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _censor_note(config: Dict[str, Any], summaries: List[Dict[str, Any]], n_terms: int) -> str:
    pieces: List[str] = []
    for mode_name in _mode_order(config):
        mode_rows = [row for row in summaries if int(row.get("n_terms") or -1) == int(n_terms) and row.get("mode_name") == mode_name]
        if not mode_rows:
            continue
        incomplete: List[str] = []
        for row in mode_rows:
            if row.get("status") != "ok":
                incomplete.append(str(row.get("status")))
                continue
            if not row.get("reached_highest_milestone"):
                final_acc = _safe_float(row.get("final_digit_accuracy"))
                highest_reached = _safe_float(row.get("highest_reached_milestone"))
                if final_acc is not None:
                    incomplete.append(f"final {final_acc * 100:.1f}%")
                elif highest_reached is not None:
                    incomplete.append(f"reached through {highest_reached * 100:.0f}%")
                else:
                    incomplete.append("no milestone reached")
        if incomplete:
            pieces.append(f"{mode_name}: {', '.join(incomplete[:3])}")
    if not pieces:
        return ""
    return "Censored / incomplete runs: " + "; ".join(pieces)


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
    percent = value * 100.0
    if abs(percent - round(percent)) < 1e-9:
        return f"{percent:.0f}%"
    return f"{percent:g}%"


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

    fig, ax = plt.subplots(figsize=(max(A4_PAGE_WIDTH_IN, 8.5, 0.75 * len(milestones) + 2.5), 5.0))
    values_for_scale: List[float] = []
    plotted_modes: List[str] = []
    draw_error_bars = _show_milestone_error_bars(config)

    for mode_idx, mode_name in enumerate(mode_names):
        by_milestone: Dict[float, List[float]] = {}
        for milestone in milestones:
            complete_rows = _fully_reached_milestone_rows(
                config,
                term_all_rows,
                mode_name=mode_name,
                milestone=milestone,
                required_fields=(metric,),
            )
            if complete_rows:
                by_milestone[milestone] = [float(row[metric]) for row in complete_rows]

        xs: List[float] = []
        ys: List[float] = []
        err_lowers: List[float] = []
        err_uppers: List[float] = []
        offset = -group_width / 2.0 + bar_width / 2.0 + mode_idx * bar_width
        for milestone_idx, milestone in enumerate(milestones):
            values = by_milestone.get(milestone)
            if not values:
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
            label=mode_name,
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
    ax.set_title(f"{n_terms}-term SPLL training: {ylabel}", loc="left")
    ax.set_xlabel("Training true-sum posterior checkpoint")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_format_milestone_label(value) for value in milestones])
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if plotted_modes:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    note = _censor_note(config, run_summaries, n_terms)
    if note:
        fig.text(0.01, 0.033, note, ha="left", va="bottom", fontsize=8)
    _add_uncertainty_note(fig, config, enabled=draw_error_bars, y=0.015)
    if note or draw_error_bars:
        fig.tight_layout(rect=(0, 0.075, 1, 1))
    else:
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
    fig, ax = plt.subplots(figsize=(max(A4_PAGE_WIDTH_IN, 8.5), 5.0))
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
        ax.plot(xs, ys, label=mode_name, linewidth=2.15, color=colors[mode_name], zorder=3)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(f"{n_terms}-term SPLL training: {ylabel}", loc="left")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_kwargs: Dict[str, Any] = {"loc": legend_loc, "frameon": False}
    if legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox
    ax.legend(**legend_kwargs)
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


def _adaptive_mode_names(config: Dict[str, Any]) -> List[str]:
    return [str(mode["name"]) for mode in get_inference_modes(config) if bool(mode.get("adaptive_top_k", False))]


def _mode_posterior_mass_target(config: Dict[str, Any], mode_name: str) -> Optional[float]:
    for mode in get_inference_modes(config):
        if str(mode["name"]) == str(mode_name):
            return _safe_float(mode.get("posterior_mass_target"))
    return None


def _add_no_smoothing_note(fig: Any, y: float = 0.015) -> None:
    fig.text(
        0.01,
        y,
        "No rolling mean: adaptive top-k values are calibrated control settings, not noisy per-batch observations.",
        ha="left",
        va="bottom",
        fontsize=8,
    )


def _plot_adaptive_topk_event_trace(
    *,
    config: Dict[str, Any],
    n_terms: int,
    value_key: str,
    ylabel: str,
    output_path: Path,
    show_target: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(A4_PAGE_WIDTH_IN, 8.5), 5.0))
    colors = _mode_color_map(config)
    plotted = False
    for mode_name in _adaptive_mode_names(config):
        xs, ys, lowers, uppers, _counts = _merged_trace_stats(
            config,
            n_terms,
            mode_name,
            "adaptive_topk_events.csv",
            value_key,
            smooth_window=1,
        )
        if not xs:
            continue
        ax.step(xs, ys, where="post", label=mode_name, linewidth=1.7, color=colors[mode_name])
        ax.scatter(xs, ys, s=18, color=colors[mode_name], zorder=3)
        if show_target:
            target = _mode_posterior_mass_target(config, mode_name)
            if target is not None:
                ax.axhline(target, color=colors[mode_name], linestyle="--", linewidth=1.0, alpha=0.65)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(f"{n_terms}-term SPLL training: adaptive top-k {ylabel}", loc="left")
    ax.set_xlabel("Training step at cutoff refresh")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    _add_no_smoothing_note(fig)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _latest_search_event_rows(
    *,
    config: Dict[str, Any],
    seed: int,
    n_terms: int,
    mode_name: str,
) -> List[Dict[str, str]]:
    paths = training_paths(config)
    rows = _read_trace_csv(run_dir(paths, seed, n_terms, mode_name) / "adaptive_topk_search_trace.csv")
    if not rows:
        return []
    valid_steps = [_safe_int(row.get("step")) for row in rows]
    latest_step = max((value for value in valid_steps if value is not None), default=None)
    if latest_step is None:
        return []
    latest_rows = [row for row in rows if _safe_int(row.get("step")) == latest_step]
    if not latest_rows:
        return []
    # If several refresh reasons happened at the same step, keep the last one
    # written in the trace. This is normally the final refresh at run end.
    latest_reason = latest_rows[-1].get("reason")
    return [row for row in latest_rows if row.get("reason") == latest_reason]


def _plot_adaptive_topk_search_iterations(
    *,
    config: Dict[str, Any],
    n_terms: int,
    value_key: str,
    ylabel: str,
    output_path: Path,
    show_target: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(A4_PAGE_WIDTH_IN, 8.5), 5.0))
    colors = _mode_color_map(config)
    plotted = False
    for mode_name in _adaptive_mode_names(config):
        by_eval: Dict[int, List[float]] = {}
        selected_cutoffs: List[float] = []
        for seed in get_seeds(config):
            event_rows = _latest_search_event_rows(config=config, seed=int(seed), n_terms=n_terms, mode_name=mode_name)
            if not event_rows:
                continue
            selected = _safe_float(event_rows[0].get("runtime_top_k_cutoff"))
            if selected is not None:
                selected_cutoffs.append(selected)
            for row in event_rows:
                eval_idx = _safe_int(row.get("evaluation_index"))
                value = _safe_float(row.get(value_key))
                if eval_idx is None or value is None:
                    continue
                by_eval.setdefault(eval_idx, []).append(value)
        if not by_eval:
            continue
        xs = sorted(by_eval)
        ys = [float(sum(by_eval[x]) / len(by_eval[x])) for x in xs]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.7, label=mode_name, color=colors[mode_name])
        if value_key == "candidate_cutoff" and selected_cutoffs:
            ax.axhline(sum(selected_cutoffs) / len(selected_cutoffs), color=colors[mode_name], linestyle="--", linewidth=1.0, alpha=0.65)
        if show_target:
            target = _mode_posterior_mass_target(config, mode_name)
            if target is not None:
                ax.axhline(target, color=colors[mode_name], linestyle="--", linewidth=1.0, alpha=0.65)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(f"{n_terms}-term SPLL training: latest adaptive top-k search iterations", loc="left")
    ax.set_xlabel("Cutoff-search evaluation index")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    _add_no_smoothing_note(fig)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)






if __name__ == "__main__":
    main()
