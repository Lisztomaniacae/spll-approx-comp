from __future__ import annotations

import csv
import json
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
                            "milestone": float(milestone),
                            "reached": bool(info.get("reached", False)),
                            "step": info.get("step"),
                            "elapsed_seconds": info.get("elapsed_seconds"),
                            "digit_accuracy": info.get("digit_accuracy"),
                        }
                    )
    return rows


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
                            "status": "failed_or_missing" if failure_path.exists() else "missing",
                            "failure_report": str(failure_path) if failure_path.exists() else "",
                            "final_step": None,
                            "final_elapsed_seconds": None,
                            "final_digit_accuracy": None,
                            "highest_milestone": None,
                            "highest_reached_milestone": None,
                            "reached_highest_milestone": False,
                            "mean_step_seconds": None,
                            "zero_true_mass_rate": None,
                            "mean_branch_count": None,
                            "final_loss": None,
                            "final_true_mass": None,
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
                        "status": "ok",
                        "failure_report": "",
                        "final_step": final_step,
                        "final_elapsed_seconds": final_elapsed,
                        "final_digit_accuracy": summary.get("final_digit_accuracy"),
                        "highest_milestone": highest,
                        "highest_reached_milestone": highest_reached,
                        "reached_highest_milestone": bool(highest is not None and highest_reached is not None and highest_reached >= highest),
                        "mean_step_seconds": (final_elapsed / final_step) if final_elapsed is not None and final_step else None,
                        "zero_true_mass_rate": (sum(zero_values) / len(zero_values)) if zero_values else None,
                        "mean_branch_count": (sum(branch_values) / len(branch_values)) if branch_values else None,
                        "final_loss": final_loss,
                        "final_true_mass": final_true_mass,
                    }
                )
    return rows


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

    for mode_idx, mode_name in enumerate(mode_names):
        mode_rows = [row for row in term_rows if row["mode_name"] == mode_name]
        by_milestone: Dict[float, List[float]] = {}
        for row in mode_rows:
            by_milestone.setdefault(float(row["milestone"]), []).append(float(row[metric]))

        xs: List[float] = []
        ys: List[float] = []
        offset = -group_width / 2.0 + bar_width / 2.0 + mode_idx * bar_width
        for milestone_idx, milestone in enumerate(milestones):
            values = by_milestone.get(milestone)
            if not values:
                continue
            xs.append(float(x_positions[milestone_idx]) + offset)
            ys.append(sum(values) / len(values))

        if not ys:
            continue
        values_for_scale.extend(ys)
        plotted_modes.append(mode_name)
        ax.bar(
            xs,
            ys,
            width=bar_width,
            label=mode_name,
            color=colors[mode_name],
            edgecolor="white",
            linewidth=0.8,
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
    ax.set_xlabel("Held-out digit accuracy milestone")
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
        fig.text(0.01, 0.015, note, ha="left", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def _rolling_mean(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return [float(v) for v in values]
    result: List[float] = []
    running: Deque[float] = deque()
    total = 0.0
    for value in values:
        value = float(value)
        running.append(value)
        total += value
        while len(running) > window:
            total -= running.popleft()
        result.append(total / len(running))
    return result


def _merged_trace_series(config: Dict[str, Any], n_terms: int, mode_name: str, trace_name: str, value_key: str) -> Tuple[List[int], List[float]]:
    paths = training_paths(config)
    merged: Dict[int, List[float]] = {}
    for seed in get_seeds(config):
        rows = _read_trace_csv(run_dir(paths, seed, n_terms, mode_name) / trace_name)
        for row in rows:
            value = _safe_float(row.get(value_key))
            step = _safe_int(row.get("step"))
            if value is None or step is None:
                continue
            merged.setdefault(step, []).append(value)
    xs = sorted(merged)
    ys = [sum(merged[x]) / len(merged[x]) for x in xs]
    return xs, ys


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
    for mode_name in _mode_order(config):
        xs, ys = _merged_trace_series(config, n_terms, mode_name, trace_name, value_key)
        if not xs:
            continue
        display_ys = _rolling_mean(ys, smooth_window)
        ax.plot(xs, display_ys, label=mode_name, linewidth=1.5, color=colors[mode_name])
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    smoothing_suffix = f" (rolling mean, {smooth_window} points)" if smooth_window > 1 else ""
    ax.set_title(f"{n_terms}-term SPLL training: {ylabel}{smoothing_suffix}", loc="left")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_visualization_stage(config: Dict[str, Any]) -> None:
    paths = training_paths(config)
    stage_message(1, 3, "Writing resolved Pipeline II config snapshot")
    save_config(config, paths.config_used_path)

    stage_message(2, 3, "Collecting milestone and run summaries")
    rows = _collect_rows(config)
    run_summaries = _collect_run_summaries(config)
    milestone_fields = [
        "seed",
        "n_terms",
        "mode_name",
        "top_k_cutoff",
        "milestone",
        "reached",
        "step",
        "elapsed_seconds",
        "digit_accuracy",
    ]
    run_summary_fields = [
        "seed",
        "n_terms",
        "mode_name",
        "top_k_cutoff",
        "status",
        "failure_report",
        "final_step",
        "final_elapsed_seconds",
        "final_digit_accuracy",
        "highest_milestone",
        "highest_reached_milestone",
        "reached_highest_milestone",
        "mean_step_seconds",
        "zero_true_mass_rate",
        "mean_branch_count",
        "final_loss",
        "final_true_mass",
    ]
    _write_csv(paths.tables_root / "milestone_summary.csv", rows, milestone_fields)
    write_json(paths.tables_root / "milestone_summary.json", rows)
    _write_csv(paths.tables_root / "run_summary.csv", run_summaries, run_summary_fields)
    write_json(paths.tables_root / "run_summary.json", run_summaries)
    print(f"Saved milestone summary to: {paths.tables_root / 'milestone_summary.csv'}")
    print(f"Saved run summary to: {paths.tables_root / 'run_summary.csv'}")

    stage_message(3, 3, "Writing Pipeline II figures")
    viz_cfg = config.get("visualization", {})
    smooth_window = int(viz_cfg.get("trace_smoothing_window_points", 100))
    for experiment in get_experiments(config):
        n_terms = int(experiment["n_terms"])
        _plot_metric_for_terms(
            config=config,
            rows=rows,
            run_summaries=run_summaries,
            n_terms=n_terms,
            metric="step",
            ylabel="Steps to milestone",
            output_path=paths.figures_main_text_root / f"terms_{n_terms:02d}_steps_to_digit_milestone.png",
        )
        _plot_metric_for_terms(
            config=config,
            rows=rows,
            run_summaries=run_summaries,
            n_terms=n_terms,
            metric="elapsed_seconds",
            ylabel="Seconds to milestone",
            output_path=paths.figures_main_text_root / f"terms_{n_terms:02d}_time_to_digit_milestone.png",
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
