from __future__ import annotations

import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from mnist_spll_common import load_config, save_config, stage_message
from mnist_spll_pipeline_core import write_json
from spll_training_core import ensure_dir, get_experiments, get_inference_modes, get_seeds, run_dir, training_paths


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> Optional[float]:
    if value in {None, "", "None"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    number = _safe_float(value)
    return int(number) if number is not None else None


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _sample_std(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _viz_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(config.get("visualization", {}) or {})


def _uncertainty_mode(config: Dict[str, Any]) -> str:
    """Return the configured across-seed uncertainty interval.

    Supported values:
    - ``std``: one sample standard deviation across seeds;
    - ``sem``: standard error of the mean;
    - ``ci95``: approximate normal 95% confidence interval around the mean;
    - ``none``: disable error bars / bands.

    ``std`` is the default because Pipeline II currently has only a handful of
    seeds; it honestly shows run-to-run spread without implying an overprecise
    population confidence interval.
    """

    mode = str(_viz_cfg(config).get("uncertainty_interval", "std")).strip().lower()
    if mode not in {"std", "sem", "ci95", "none"}:
        raise ValueError(
            "visualization.uncertainty_interval must be one of: "
            "std, sem, ci95, none"
        )
    return mode


def _min_uncertainty_samples(config: Dict[str, Any]) -> int:
    return max(2, int(_viz_cfg(config).get("min_uncertainty_samples", 2)))


def _uncertainty_half_width(values: Sequence[float], config: Dict[str, Any]) -> Optional[float]:
    mode = _uncertainty_mode(config)
    if mode == "none" or len(values) < _min_uncertainty_samples(config):
        return None
    std = _sample_std(values)
    if std is None:
        return None
    if mode == "std":
        return std
    sem = std / math.sqrt(len(values))
    if mode == "sem":
        return sem
    return 1.96 * sem


def _uncertainty_label(config: Dict[str, Any]) -> str:
    mode = _uncertainty_mode(config)
    labels = {
        "std": "±1 sample std over seeds",
        "sem": "±1 SEM over seeds",
        "ci95": "approx. 95% CI over seeds",
        "none": "uncertainty disabled",
    }
    return labels[mode]


def _show_milestone_error_bars(config: Dict[str, Any]) -> bool:
    return bool(_viz_cfg(config).get("show_milestone_error_bars", True))


def _show_trace_uncertainty_bands(config: Dict[str, Any], smooth_window: int) -> bool:
    viz_cfg = _viz_cfg(config)
    if _uncertainty_mode(config) == "none":
        return False
    if smooth_window <= 1 and not bool(viz_cfg.get("show_raw_trace_uncertainty_bands", False)):
        return False
    return bool(viz_cfg.get("show_trace_uncertainty_bands", True))


def _trace_band_alpha(config: Dict[str, Any]) -> float:
    return max(0.0, min(1.0, float(_viz_cfg(config).get("trace_band_alpha", 0.16))))


def _add_uncertainty_note(fig: Any, config: Dict[str, Any], *, enabled: bool, y: float = 0.015) -> None:
    if not enabled or _uncertainty_mode(config) == "none":
        return
    fig.text(
        0.01,
        y,
        f"Mean over configured seeds; uncertainty shows {_uncertainty_label(config)}.",
        ha="left",
        va="bottom",
        fontsize=8,
    )


def _collect_rows(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    paths = training_paths(config)
    rows: List[Dict[str, Any]] = []
    for seed in get_seeds(config):
        for experiment in get_experiments(config):
            n_terms = int(experiment["n_terms"])
            for mode in get_inference_modes(config):
                mode_name = str(mode["name"])
                summary_path = run_dir(paths, seed, n_terms, mode_name) / "run_summary.json"
                if not summary_path.exists():
                    continue
                summary = _read_json(summary_path)
                for milestone, info in summary.get("milestones", {}).items():
                    rows.append(
                        {
                            "seed": seed,
                            "n_terms": n_terms,
                            "mode_name": mode_name,
                            "top_k_cutoff": mode.get("top_k_cutoff"),
                            "adaptive_top_k": bool(mode.get("adaptive_top_k", False)),
                            "posterior_mass_target": mode.get("posterior_mass_target"),
                            "runtime_top_k_cutoff": summary.get("runtime_top_k_cutoff"),
                            "milestone": float(milestone),
                            "reached": bool(info.get("reached", False)),
                            "step": info.get("step"),
                            "elapsed_seconds": info.get("elapsed_seconds"),
                            "digit_accuracy": info.get("digit_accuracy"),
                            "sum_posterior_accuracy": info.get("sum_posterior_accuracy", info.get("digit_accuracy")),
                            "validation_metric": info.get("validation_metric", "sum_posterior_accuracy"),
                        }
                    )
    return rows


def _metric_summary(values: Sequence[float], config: Dict[str, Any]) -> Dict[str, Optional[float]]:
    mean_value = _mean(values)
    std_value = _sample_std(values)
    half_width = _uncertainty_half_width(values, config)
    if not values:
        return {
            "mean": None,
            "std": None,
            "uncertainty_half_width": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": mean_value,
        "std": std_value,
        "uncertainty_half_width": half_width,
        "min": min(values),
        "max": max(values),
    }


def _collect_milestone_aggregates(config: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str, float], List[Dict[str, Any]]] = {}
    configured_seed_count = len(get_seeds(config))
    for row in rows:
        key = (int(row["n_terms"]), str(row["mode_name"]), float(row["milestone"]))
        grouped.setdefault(key, []).append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item[0], _mode_order(config).index(item[1]) if item[1] in _mode_order(config) else 999, item[2])):
        n_terms, mode_name, milestone = key
        group_rows = grouped[key]
        reached_rows = [row for row in group_rows if row.get("reached")]
        step_values = [float(row["step"]) for row in reached_rows if row.get("step") not in {None, ""}]
        time_values = [float(row["elapsed_seconds"]) for row in reached_rows if row.get("elapsed_seconds") not in {None, ""}]
        accuracy_values = [float(row["digit_accuracy"]) for row in reached_rows if row.get("digit_accuracy") not in {None, ""}]
        step_summary = _metric_summary(step_values, config)
        time_summary = _metric_summary(time_values, config)
        accuracy_summary = _metric_summary(accuracy_values, config)
        aggregate_rows.append(
            {
                "n_terms": n_terms,
                "mode_name": mode_name,
                "top_k_cutoff": group_rows[0].get("top_k_cutoff") if group_rows else None,
                "adaptive_top_k": group_rows[0].get("adaptive_top_k") if group_rows else None,
                "posterior_mass_target": group_rows[0].get("posterior_mass_target") if group_rows else None,
                "runtime_top_k_cutoff_mean": _mean([float(row["runtime_top_k_cutoff"]) for row in group_rows if row.get("runtime_top_k_cutoff") not in {None, ""}]),
                "milestone": milestone,
                "configured_seed_count": configured_seed_count,
                "observed_seed_count": len({int(row["seed"]) for row in group_rows}),
                "reached_seed_count": len({int(row["seed"]) for row in reached_rows}),
                "step_mean": step_summary["mean"],
                "step_std": step_summary["std"],
                "step_uncertainty_half_width": step_summary["uncertainty_half_width"],
                "step_min": step_summary["min"],
                "step_max": step_summary["max"],
                "elapsed_seconds_mean": time_summary["mean"],
                "elapsed_seconds_std": time_summary["std"],
                "elapsed_seconds_uncertainty_half_width": time_summary["uncertainty_half_width"],
                "elapsed_seconds_min": time_summary["min"],
                "elapsed_seconds_max": time_summary["max"],
                "digit_accuracy_mean": accuracy_summary["mean"],
                "digit_accuracy_std": accuracy_summary["std"],
                "digit_accuracy_uncertainty_half_width": accuracy_summary["uncertainty_half_width"],
            }
        )
    return aggregate_rows


def _read_trace_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mode_order(config: Dict[str, Any]) -> List[str]:
    return [str(mode["name"]) for mode in get_inference_modes(config)]


def _mode_color_map(config: Dict[str, Any]) -> Dict[str, str]:
    """Return a stable, colorblind-friendly color per inference mode.

    The mapping is derived from config order so the same mode keeps the same
    color in milestone bar charts and trace figures, independent of which
    milestones were reached in a particular run.
    """

    palette = [
        "#4E79A7",
        "#F28E2B",
        "#59A14F",
        "#E15759",
        "#B07AA1",
        "#76B7B2",
        "#EDC948",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]
    return {mode_name: palette[idx % len(palette)] for idx, mode_name in enumerate(_mode_order(config))}


def _highest_milestone(summary: Dict[str, Any]) -> Optional[float]:
    milestones = summary.get("milestones", {})
    if not milestones:
        return None
    return max(float(value) for value in milestones.keys())


def _highest_reached_milestone(summary: Dict[str, Any]) -> Optional[float]:
    reached = [float(value) for value, info in summary.get("milestones", {}).items() if info.get("reached")]
    return max(reached) if reached else None


def _collect_run_summaries(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    paths = training_paths(config)
    rows: List[Dict[str, Any]] = []
    for seed in get_seeds(config):
        for experiment in get_experiments(config):
            n_terms = int(experiment["n_terms"])
            for mode in get_inference_modes(config):
                mode_name = str(mode["name"])
                this_run_dir = run_dir(paths, seed, n_terms, mode_name)
                summary_path = this_run_dir / "run_summary.json"
                if not summary_path.exists():
                    failure_path = this_run_dir / "failure_report.json"
                    rows.append(
                        {
                            "seed": seed,
                            "n_terms": n_terms,
                            "mode_name": mode_name,
                            "top_k_cutoff": mode.get("top_k_cutoff"),
                            "adaptive_top_k": bool(mode.get("adaptive_top_k", False)),
                            "posterior_mass_target": mode.get("posterior_mass_target"),
                            "runtime_top_k_cutoff": None,
                            "status": "failed_or_missing" if failure_path.exists() else "missing",
                            "failure_report": str(failure_path) if failure_path.exists() else "",
                            "final_step": None,
                            "final_elapsed_seconds": None,
                            "final_digit_accuracy": None,
                            "final_sum_posterior_accuracy": None,
                            "validation_metric": None,
                            "highest_milestone": None,
                            "highest_reached_milestone": None,
                            "reached_highest_milestone": False,
                            "mean_step_seconds": None,
                            "zero_true_mass_rate": None,
                            "mean_branch_count": None,
                            "final_loss": None,
                            "final_true_mass": None,
                            "final_loss_recent_mean": None,
                            "final_true_mass_recent_mean": None,
                        }
                    )
                    continue

                summary = _read_json(summary_path)
                train_rows = _read_trace_csv(this_run_dir / "train_trace.csv")
                final_step: Optional[int] = None
                final_loss: Optional[float] = None
                final_true_mass: Optional[float] = None
                zero_values: List[float] = []
                branch_values: List[float] = []
                for trace_row in train_rows:
                    step = _safe_int(trace_row.get("step"))
                    if step is not None and (final_step is None or step > final_step):
                        final_step = step
                        final_loss = _safe_float(trace_row.get("loss"))
                        final_true_mass = _safe_float(trace_row.get("true_mass"))
                    zero = _safe_float(trace_row.get("zero_true_mass"))
                    branch = _safe_float(trace_row.get("branch_count"))
                    if zero is not None:
                        zero_values.append(zero)
                    if branch is not None:
                        branch_values.append(branch)

                final_elapsed = _safe_float(summary.get("final_elapsed_seconds"))
                highest = _highest_milestone(summary)
                highest_reached = _highest_reached_milestone(summary)
                rows.append(
                    {
                        "seed": int(seed),
                        "n_terms": n_terms,
                        "mode_name": mode_name,
                        "top_k_cutoff": mode.get("top_k_cutoff"),
                        "adaptive_top_k": bool(summary.get("adaptive_top_k", mode.get("adaptive_top_k", False))),
                        "posterior_mass_target": summary.get("posterior_mass_target", mode.get("posterior_mass_target")),
                        "runtime_top_k_cutoff": summary.get("runtime_top_k_cutoff"),
                        "status": "ok",
                        "failure_report": "",
                        "final_step": final_step,
                        "final_elapsed_seconds": final_elapsed,
                        "final_digit_accuracy": summary.get("final_digit_accuracy"),
                        "final_sum_posterior_accuracy": summary.get("final_sum_posterior_accuracy", summary.get("final_digit_accuracy")),
                        "validation_metric": summary.get("validation_metric", "sum_posterior_accuracy"),
                        "highest_milestone": highest,
                        "highest_reached_milestone": highest_reached,
                        "reached_highest_milestone": bool(highest is not None and highest_reached is not None and highest_reached >= highest),
                        "mean_step_seconds": (final_elapsed / final_step) if final_elapsed is not None and final_step else None,
                        "zero_true_mass_rate": (sum(zero_values) / len(zero_values)) if zero_values else None,
                        "mean_branch_count": (sum(branch_values) / len(branch_values)) if branch_values else None,
                        "final_loss": final_loss,
                        "final_true_mass": final_true_mass,
                        "final_loss_recent_mean": summary.get("final_loss_recent_mean"),
                        "final_true_mass_recent_mean": summary.get("final_true_mass_recent_mean"),
                    }
                )
    return rows


def _collect_adaptive_topk_trace_rows(config: Dict[str, Any], filename: str) -> List[Dict[str, Any]]:
    paths = training_paths(config)
    rows: List[Dict[str, Any]] = []
    for seed in get_seeds(config):
        for experiment in get_experiments(config):
            n_terms = int(experiment["n_terms"])
            for mode in get_inference_modes(config):
                if not bool(mode.get("adaptive_top_k", False)):
                    continue
                mode_name = str(mode["name"])
                trace_path = run_dir(paths, seed, n_terms, mode_name) / filename
                for row in _read_trace_csv(trace_path):
                    enriched = dict(row)
                    enriched.update(
                        {
                            "seed": int(seed),
                            "n_terms": int(n_terms),
                            "mode_name": mode_name,
                            "posterior_mass_target_config": mode.get("posterior_mass_target"),
                        }
                    )
                    rows.append(enriched)
    return rows


def _checkpoint_transfer_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = config.get("checkpoint_transfer", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("checkpoint_transfer must be a mapping when provided.")
    return {
        "enabled": bool(raw.get("enabled", True)),
        "anchor_mode_name": str(raw.get("anchor_mode_name", "exact")).strip() or "exact",
        "mode_names": raw.get("mode_names"),
    }


def _checkpoint_transfer_run_dir(paths: Any, seed: int, n_terms: int, mode_name: str, anchor_mode_name: str) -> Path:
    return paths.runs_root / f"seed_{int(seed)}_terms_{int(n_terms):02d}_transfer_{mode_name}_from_{anchor_mode_name}"


def _aggregate_checkpoint_path(paths: Any, n_terms: int, anchor_mode_name: str) -> Path:
    return paths.root / "aggregate_checkpoints" / f"terms_{int(n_terms):02d}_{anchor_mode_name}_posterior_checkpoints.json"


def _collect_checkpoint_transfer_rows(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = _checkpoint_transfer_cfg(config)
    if not cfg.get("enabled", True):
        return []
    anchor_mode_name = str(cfg["anchor_mode_name"])
    requested_mode_names = cfg.get("mode_names")
    allowed = None if requested_mode_names is None else {str(v) for v in requested_mode_names}
    paths = training_paths(config)
    rows: List[Dict[str, Any]] = []
    for seed in get_seeds(config):
        for experiment in get_experiments(config):
            n_terms = int(experiment["n_terms"])
            for mode in get_inference_modes(config):
                mode_name = str(mode["name"])
                if mode_name == anchor_mode_name:
                    continue
                if allowed is not None and mode_name not in allowed:
                    continue
                trace_path = _checkpoint_transfer_run_dir(paths, seed, n_terms, mode_name, anchor_mode_name) / "checkpoint_transfer_trace.csv"
                for row in _read_trace_csv(trace_path):
                    enriched = dict(row)
                    enriched.update(
                        {
                            "seed": int(seed),
                            "n_terms": int(n_terms),
                            "mode_name": mode_name,
                            "anchor_mode_name": anchor_mode_name,
                        }
                    )
                    rows.append(enriched)
    return rows


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


def _checkpoint_transfer_anchor_metric_key(value_key: str) -> Optional[str]:
    if value_key in {"true_mass", "true_mass_recent_mean"}:
        return "anchor_rolling_true_mass_exact"
    if value_key in {"loss", "loss_recent_mean"}:
        return "anchor_rolling_loss_exact"
    return None


def _checkpoint_transfer_target_metric_key(value_key: str) -> Optional[str]:
    if value_key in {"true_mass", "true_mass_recent_mean"}:
        return "target_rolling_true_mass_exact"
    if value_key in {"loss", "loss_recent_mean"}:
        return "target_rolling_loss_exact"
    return None


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
    by_segment: Dict[int, Dict[int, Tuple[List[float], List[float]]]] = {}
    for seed in get_seeds(config):
        this_dir = _checkpoint_transfer_run_dir(paths, seed, n_terms, mode_name, anchor_mode_name)
        # Segment metadata is read by the summary tables.  The plotted green
        # curve intentionally uses only approximate-training observations, not
        # the exact anchor value, so each full-window point represents 50
        # approximate updates after the restart.
        _segment_rows = _read_trace_csv(this_dir / "checkpoint_transfer_trace.csv")

        train_rows = _read_trace_csv(this_dir / "checkpoint_transfer_train_trace.csv")
        per_segment_step: Dict[int, Dict[int, List[float]]] = {}
        for row in train_rows:
            segment_index = _safe_int(row.get("segment_index"))
            step = _safe_int(row.get("step"))
            value = _safe_float(row.get(value_key))
            if segment_index is None or step is None or value is None:
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
            # Prefix points shorter than the full window are omitted, so segment
            # starts cannot create unstable short-window spikes.
            rolled_xs, rolled_ys = _full_window_rolling_series(xs, ys, smooth_window)
            if not rolled_xs:
                continue
            by_segment.setdefault(segment_index, {})[int(seed)] = (rolled_xs, rolled_ys)

    return [by_segment[index] for index in sorted(by_segment)]


def _exact_posterior_checkpoint_markers(
    config: Dict[str, Any],
    n_terms: int,
    anchor_mode_name: str,
    value_key: str,
) -> Tuple[List[float], List[float]]:
    paths = training_paths(config)
    aggregate_path = _aggregate_checkpoint_path(paths, n_terms, anchor_mode_name)
    if not aggregate_path.exists():
        return [], []
    payload = _read_json(aggregate_path)
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
    return xs, ys

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

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    colors = {
        "exact": "#3776d6",
        "pure": "#d62728",
        "transfer": "#2ca02c",
        "checkpoint": "#7b3294",
    }
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

    checkpoint_x, checkpoint_y = _exact_posterior_checkpoint_markers(config, n_terms, anchor_mode_name, value_key)
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

    smoothing_suffix = f" (full-window rolling mean, {smooth_window} updates)" if smooth_window > 1 else ""
    ax.set_title(f"{n_terms}-term SPLL training: {ylabel}{smoothing_suffix}", loc="left")
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
        "Purple X markers are aggregate exact checkpoints computed from the same full-window rolling curve. Green pieces are separate approximate continuations; first short-window points in each segment are omitted.",
        ha="left",
        va="bottom",
        fontsize=8,
    )
    _add_uncertainty_note(fig, config, enabled=draw_bands, y=0.034)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
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
    term_rows = [row for row in term_all_rows if row["reached"] and row[metric] not in {None, ""}]
    if not term_rows:
        return

    milestones = sorted({float(row["milestone"]) for row in term_all_rows})
    mode_names = _mode_order(config)
    colors = _mode_color_map(config)
    x_positions = list(range(len(milestones)))
    mode_count = max(1, len(mode_names))
    bar_width = min(0.82 / mode_count, 0.18)
    group_width = bar_width * mode_count

    fig, ax = plt.subplots(figsize=(max(8.5, 0.75 * len(milestones) + 2.5), 5.0))
    values_for_scale: List[float] = []
    plotted_modes: List[str] = []
    draw_error_bars = _show_milestone_error_bars(config)

    for mode_idx, mode_name in enumerate(mode_names):
        mode_rows = [row for row in term_rows if row["mode_name"] == mode_name]
        by_milestone: Dict[float, List[float]] = {}
        for row in mode_rows:
            by_milestone.setdefault(float(row["milestone"]), []).append(float(row[metric]))

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
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
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
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
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
        ax.plot(xs, ys, label=mode_name, linewidth=1.5, color=colors[mode_name])
        if draw_bands and any(abs(upper - lower) > 0.0 for lower, upper in zip(lowers, uppers)):
            ax.fill_between(xs, lowers, uppers, color=colors[mode_name], alpha=_trace_band_alpha(config), linewidth=0)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    smoothing_suffix = f" (full-window rolling mean, {smooth_window} points)" if smooth_window > 1 else ""
    ax.set_title(f"{n_terms}-term SPLL training: {ylabel}{smoothing_suffix}", loc="left")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    _add_uncertainty_note(fig, config, enabled=draw_bands)
    if draw_bands:
        fig.tight_layout(rect=(0, 0.045, 1, 1))
    else:
        fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
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
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
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
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
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
    # If several refresh reasons happened at the same step, keep the last one as
    # written in the trace. This is usually final_validation at run end.
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
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
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
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_visualization_stage(config: Dict[str, Any]) -> None:
    paths = training_paths(config)
    stage_message(1, 3, "Writing resolved Pipeline II config snapshot")
    save_config(config, paths.config_used_path)

    stage_message(2, 3, "Collecting training, checkpoint, and run summaries")
    rows = _collect_rows(config)
    milestone_aggregates = _collect_milestone_aggregates(config, rows)
    run_summaries = _collect_run_summaries(config)
    adaptive_topk_events = _collect_adaptive_topk_trace_rows(config, "adaptive_topk_events.csv")
    adaptive_topk_search_rows = _collect_adaptive_topk_trace_rows(config, "adaptive_topk_search_trace.csv")
    checkpoint_transfer_rows = _collect_checkpoint_transfer_rows(config)
    milestone_fields = [
        "seed",
        "n_terms",
        "mode_name",
        "top_k_cutoff",
        "adaptive_top_k",
        "posterior_mass_target",
        "runtime_top_k_cutoff",
        "milestone",
        "reached",
        "step",
        "elapsed_seconds",
        "digit_accuracy",
        "sum_posterior_accuracy",
        "validation_metric",
    ]
    run_summary_fields = [
        "seed",
        "n_terms",
        "mode_name",
        "top_k_cutoff",
        "adaptive_top_k",
        "posterior_mass_target",
        "runtime_top_k_cutoff",
        "status",
        "failure_report",
        "final_step",
        "final_elapsed_seconds",
        "final_digit_accuracy",
        "final_sum_posterior_accuracy",
        "validation_metric",
        "highest_milestone",
        "highest_reached_milestone",
        "reached_highest_milestone",
        "mean_step_seconds",
        "zero_true_mass_rate",
        "mean_branch_count",
        "final_loss",
        "final_true_mass",
        "final_loss_recent_mean",
        "final_true_mass_recent_mean",
    ]
    milestone_aggregate_fields = [
        "n_terms",
        "mode_name",
        "top_k_cutoff",
        "adaptive_top_k",
        "posterior_mass_target",
        "runtime_top_k_cutoff_mean",
        "milestone",
        "configured_seed_count",
        "observed_seed_count",
        "reached_seed_count",
        "step_mean",
        "step_std",
        "step_uncertainty_half_width",
        "step_min",
        "step_max",
        "elapsed_seconds_mean",
        "elapsed_seconds_std",
        "elapsed_seconds_uncertainty_half_width",
        "elapsed_seconds_min",
        "elapsed_seconds_max",
        "digit_accuracy_mean",
        "digit_accuracy_std",
        "digit_accuracy_uncertainty_half_width",
    ]
    _write_csv(paths.tables_root / "milestone_summary.csv", rows, milestone_fields)
    write_json(paths.tables_root / "milestone_summary.json", rows)
    _write_csv(paths.tables_root / "milestone_aggregate_summary.csv", milestone_aggregates, milestone_aggregate_fields)
    write_json(paths.tables_root / "milestone_aggregate_summary.json", milestone_aggregates)
    _write_csv(paths.tables_root / "run_summary.csv", run_summaries, run_summary_fields)
    write_json(paths.tables_root / "run_summary.json", run_summaries)
    _write_csv(
        paths.tables_root / "adaptive_topk_events.csv",
        adaptive_topk_events,
        [
            "seed",
            "n_terms",
            "mode_name",
            "step",
            "reason",
            "runtime_top_k_cutoff",
            "posterior_mass_target",
            "mean_surviving_posterior_mass",
            "abs_error",
            "iterations",
            "converged",
            "probe_cases",
            "search_runtime_sec",
            "evaluation_count",
        ],
    )
    _write_csv(
        paths.tables_root / "adaptive_topk_search_trace.csv",
        adaptive_topk_search_rows,
        [
            "seed",
            "n_terms",
            "mode_name",
            "step",
            "reason",
            "evaluation_index",
            "candidate_cutoff",
            "mean_surviving_mass",
            "posterior_mass_target",
            "runtime_top_k_cutoff",
            "selected_mean_surviving_posterior_mass",
            "selected_abs_error",
            "iterations",
            "converged",
        ],
    )
    _write_csv(
        paths.tables_root / "checkpoint_transfer_summary.csv",
        checkpoint_transfer_rows,
        [
            "seed",
            "n_terms",
            "mode_name",
            "anchor_mode_name",
            "segment_index",
            "anchor_label",
            "anchor_step",
            "anchor_rolling_true_mass_exact",
            "anchor_rolling_loss_exact",
            "target_label",
            "target_step",
            "target_rolling_true_mass_exact",
            "target_rolling_loss_exact",
            "segment_cases",
            "segment_elapsed_seconds",
            "end_loss_transfer",
            "end_true_mass_transfer",
            "end_loss_recent_mean_transfer",
            "end_true_mass_recent_mean_transfer",
            "end_zero_true_mass_recent_rate_transfer",
            "top_k_cutoff_runtime",
            "posterior_mass_target",
        ],
    )
    write_json(paths.tables_root / "adaptive_topk_events.json", adaptive_topk_events)
    write_json(paths.tables_root / "adaptive_topk_search_trace.json", adaptive_topk_search_rows)
    write_json(paths.tables_root / "checkpoint_transfer_summary.json", checkpoint_transfer_rows)
    print(f"Saved milestone summary to: {paths.tables_root / 'milestone_summary.csv'}")
    print(f"Saved milestone aggregate summary to: {paths.tables_root / 'milestone_aggregate_summary.csv'}")
    print(f"Saved run summary to: {paths.tables_root / 'run_summary.csv'}")
    print(f"Saved checkpoint-transfer summary to: {paths.tables_root / 'checkpoint_transfer_summary.csv'}")

    stage_message(3, 3, "Writing Pipeline II figures")
    viz_cfg = config.get("visualization", {})
    smooth_window = int(viz_cfg.get("trace_smoothing_window_points", 50))
    for experiment in get_experiments(config):
        n_terms = int(experiment["n_terms"])
        _plot_metric_for_terms(
            config=config,
            rows=rows,
            run_summaries=run_summaries,
            n_terms=n_terms,
            metric="step",
            ylabel="Steps to posterior checkpoint",
            output_path=paths.figures_main_text_root / f"terms_{n_terms:02d}_steps_to_true_sum_posterior_checkpoint.png",
        )
        _plot_metric_for_terms(
            config=config,
            rows=rows,
            run_summaries=run_summaries,
            n_terms=n_terms,
            metric="elapsed_seconds",
            ylabel="Seconds to posterior checkpoint",
            output_path=paths.figures_main_text_root / f"terms_{n_terms:02d}_time_to_true_sum_posterior_checkpoint.png",
            maybe_log=True,
        )
        _plot_trace(
            config=config,
            n_terms=n_terms,
            trace_name="train_trace.csv",
            value_key="loss",
            ylabel="Training loss",
            output_path=paths.figures_main_text_root / f"terms_{n_terms:02d}_training_loss_trace.png",
            smooth_window=smooth_window,
        )
        _plot_trace(
            config=config,
            n_terms=n_terms,
            trace_name="train_trace.csv",
            value_key="loss",
            ylabel="Training loss",
            output_path=paths.figures_appendix_root / f"terms_{n_terms:02d}_training_loss_raw_trace.png",
            smooth_window=1,
        )
        _plot_trace(
            config=config,
            n_terms=n_terms,
            trace_name="train_trace.csv",
            value_key="true_mass",
            ylabel="True-sum mass",
            output_path=paths.figures_main_text_root / f"terms_{n_terms:02d}_true_mass_trace.png",
            smooth_window=smooth_window,
        )
        _plot_trace(
            config=config,
            n_terms=n_terms,
            trace_name="train_trace.csv",
            value_key="true_mass",
            ylabel="True-sum mass",
            output_path=paths.figures_appendix_root / f"terms_{n_terms:02d}_true_mass_raw_trace.png",
            smooth_window=1,
        )
        checkpoint_transfer_cfg = _checkpoint_transfer_cfg(config)
        anchor_mode_name = str(checkpoint_transfer_cfg.get("anchor_mode_name", "exact"))
        requested_transfer_modes = checkpoint_transfer_cfg.get("mode_names")
        for mode in get_inference_modes(config):
            mode_name = str(mode["name"])
            if mode_name == anchor_mode_name:
                continue
            if isinstance(requested_transfer_modes, list) and mode_name not in {str(v) for v in requested_transfer_modes}:
                continue
            _plot_checkpoint_transfer_metric_trajectory(
                config=config,
                n_terms=n_terms,
                mode_name=mode_name,
                anchor_mode_name=anchor_mode_name,
                value_key="loss",
                ylabel="Training loss",
                output_path=paths.figures_main_text_root / f"terms_{n_terms:02d}_loss_exact_vs_{mode_name}.png",
                smooth_window=smooth_window,
            )
            _plot_checkpoint_transfer_metric_trajectory(
                config=config,
                n_terms=n_terms,
                mode_name=mode_name,
                anchor_mode_name=anchor_mode_name,
                value_key="true_mass",
                ylabel="True-sum posterior (%)",
                output_path=paths.figures_main_text_root / f"terms_{n_terms:02d}_true_mass_exact_vs_{mode_name}.png",
                smooth_window=smooth_window,
                as_percent=True,
            )
        _plot_adaptive_topk_event_trace(
            config=config,
            n_terms=n_terms,
            value_key="runtime_top_k_cutoff",
            ylabel="runtime TOP_K_CUTOFF",
            output_path=paths.figures_main_text_root / f"terms_{n_terms:02d}_adaptive_topk_runtime_cutoff_trace.png",
        )
        _plot_adaptive_topk_event_trace(
            config=config,
            n_terms=n_terms,
            value_key="mean_surviving_posterior_mass",
            ylabel="mean surviving posterior mass",
            output_path=paths.figures_main_text_root / f"terms_{n_terms:02d}_adaptive_topk_surviving_mass_trace.png",
            show_target=True,
            ylim=(0.0, 1.05),
        )
        _plot_adaptive_topk_search_iterations(
            config=config,
            n_terms=n_terms,
            value_key="candidate_cutoff",
            ylabel="candidate TOP_K_CUTOFF",
            output_path=paths.figures_appendix_root / f"terms_{n_terms:02d}_adaptive_topk_search_cutoff_iterations.png",
        )
        _plot_adaptive_topk_search_iterations(
            config=config,
            n_terms=n_terms,
            value_key="mean_surviving_mass",
            ylabel="mean surviving posterior mass",
            output_path=paths.figures_appendix_root / f"terms_{n_terms:02d}_adaptive_topk_search_mass_iterations.png",
            show_target=True,
            ylim=(0.0, 1.05),
        )
        _plot_trace(
            config=config,
            n_terms=n_terms,
            trace_name="train_trace.csv",
            value_key="branch_count",
            ylabel="Branch count",
            output_path=paths.figures_appendix_root / f"terms_{n_terms:02d}_branch_count_trace.png",
            smooth_window=smooth_window,
        )
        _plot_trace(
            config=config,
            n_terms=n_terms,
            trace_name="train_trace.csv",
            value_key="branch_count",
            ylabel="Branch count",
            output_path=paths.figures_appendix_root / f"terms_{n_terms:02d}_branch_count_raw_trace.png",
            smooth_window=1,
        )
    print(f"Saved figures under: {paths.figures_root}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Pipeline II SPLL training results.")
    parser.add_argument("--config", required=True, help="Path to the Pipeline II YAML config.")
    args = parser.parse_args()
    run_visualization_stage(load_config(args.config))


if __name__ == "__main__":
    main()
