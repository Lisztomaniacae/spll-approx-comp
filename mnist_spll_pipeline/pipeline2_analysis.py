from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pipeline2_config import (
    checkpoint_transfer_run_dir,
    get_checkpoint_transfer_config,
    get_experiments,
    get_inference_modes,
    get_seeds,
    run_dir,
    training_paths,
)
from pipeline_support import ensure_dir, load_json


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
                summary = load_json(summary_path)
                for milestone, info in summary.get("milestones", {}).items():
                    rows.append(
                        {
                            "seed": seed,
                            "n_terms": n_terms,
                            "mode_name": mode_name,
                            "top_k_cutoff": mode.get("top_k_cutoff"),
                            "runtime_top_k_cutoff": summary.get("runtime_top_k_cutoff"),
                            "milestone": float(milestone),
                            "reached": bool(info.get("reached", False)),
                            "step": info.get("step"),
                            "elapsed_seconds": info.get("elapsed_seconds"),
                            "digit_accuracy": info.get("digit_accuracy"),
                            "sum_posterior_accuracy": info.get("sum_posterior_accuracy", info.get("digit_accuracy")),
                            "validation_metric": info.get("validation_metric", "sum_posterior_accuracy"),
                            "cumulative_read_mnist_model_evaluations": info.get(
                                "cumulative_read_mnist_model_evaluations"
                            ),
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
        evaluation_values = [
            float(row["cumulative_read_mnist_model_evaluations"])
            for row in reached_rows
            if row.get("cumulative_read_mnist_model_evaluations") not in {None, ""}
        ]
        step_summary = _metric_summary(step_values, config)
        time_summary = _metric_summary(time_values, config)
        accuracy_summary = _metric_summary(accuracy_values, config)
        evaluation_summary = _metric_summary(evaluation_values, config)
        aggregate_rows.append(
            {
                "n_terms": n_terms,
                "mode_name": mode_name,
                "top_k_cutoff": group_rows[0].get("top_k_cutoff") if group_rows else None,
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
                "cumulative_read_mnist_model_evaluations_mean": evaluation_summary["mean"],
                "cumulative_read_mnist_model_evaluations_std": evaluation_summary["std"],
                "cumulative_read_mnist_model_evaluations_uncertainty_half_width": evaluation_summary["uncertainty_half_width"],
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


def _mode_cfg(config: Dict[str, Any], mode_name: str) -> Dict[str, Any]:
    for mode in get_inference_modes(config):
        if str(mode.get("name")) == str(mode_name):
            return dict(mode)
    return {}


def _non_anchor_mode_names(config: Dict[str, Any], anchor_mode_name: str) -> List[str]:
    mode_names = [mode_name for mode_name in _mode_order(config) if mode_name != anchor_mode_name]

    def _sort_key(mode_name: str) -> Tuple[float, int]:
        mode = _mode_cfg(config, mode_name)
        cutoff = _safe_float(mode.get("top_k_cutoff"))
        if cutoff is None:
            return (float("inf"), mode_names.index(mode_name))
        return (cutoff, mode_names.index(mode_name))

    return sorted(mode_names, key=_sort_key)


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
                            "runtime_top_k_cutoff": None,
                            "read_mnist_policy": None,
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
                            "final_cumulative_read_mnist_calls": None,
                            "final_cumulative_read_mnist_model_evaluations": None,
                        }
                    )
                    continue

                summary = load_json(summary_path)
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
                        "runtime_top_k_cutoff": summary.get("runtime_top_k_cutoff"),
                        "read_mnist_policy": summary.get("read_mnist_policy"),
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
                        "final_cumulative_read_mnist_calls": summary.get("final_cumulative_read_mnist_calls"),
                        "final_cumulative_read_mnist_model_evaluations": summary.get(
                            "final_cumulative_read_mnist_model_evaluations"
                        ),
                    }
                )
    return rows


def _collect_checkpoint_transfer_rows(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = get_checkpoint_transfer_config(config)
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
                trace_path = checkpoint_transfer_run_dir(paths, seed, n_terms, mode_name, anchor_mode_name) / "checkpoint_transfer_trace.csv"
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
