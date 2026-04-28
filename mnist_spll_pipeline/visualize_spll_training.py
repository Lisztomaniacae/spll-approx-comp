from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt

from mnist_spll_common import load_config, save_config, stage_message
from mnist_spll_pipeline_core import load_json, write_json
from spll_training_core import ensure_dir, get_experiments, get_inference_modes, get_seeds, run_dir, training_paths


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mode_order(config: Dict[str, Any]) -> List[str]:
    return [str(mode["name"]) for mode in get_inference_modes(config)]


def _plot_metric_for_terms(
    *,
    config: Dict[str, Any],
    rows: List[Dict[str, Any]],
    n_terms: int,
    metric: str,
    ylabel: str,
    output_path: Path,
    maybe_log: bool = False,
) -> None:
    term_rows = [row for row in rows if int(row["n_terms"]) == int(n_terms) and row["reached"] and row[metric] not in {None, ""}]
    if not term_rows:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    values_for_scale: List[float] = []
    for mode_name in _mode_order(config):
        mode_rows = [row for row in term_rows if row["mode_name"] == mode_name]
        if not mode_rows:
            continue
        by_milestone: Dict[float, List[float]] = {}
        for row in mode_rows:
            by_milestone.setdefault(float(row["milestone"]), []).append(float(row[metric]))
        xs = sorted(by_milestone)
        ys = [sum(by_milestone[x]) / len(by_milestone[x]) for x in xs]
        values_for_scale.extend(ys)
        ax.plot([x * 100.0 for x in xs], ys, marker="o", label=mode_name)

    if maybe_log:
        positive = [v for v in values_for_scale if v > 0]
        if positive and max(positive) / min(positive) > 5.0:
            ax.set_yscale("log")
    ax.set_title(f"{n_terms}-term SPLL training: {ylabel}", loc="left")
    ax.set_xlabel("Held-out digit accuracy milestone (%)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _read_trace_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_trace(
    *,
    config: Dict[str, Any],
    n_terms: int,
    trace_name: str,
    value_key: str,
    ylabel: str,
    output_path: Path,
) -> None:
    paths = training_paths(config)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    plotted = False
    for mode in get_inference_modes(config):
        mode_name = str(mode["name"])
        merged: Dict[int, List[float]] = {}
        for seed in get_seeds(config):
            rows = _read_trace_csv(run_dir(paths, seed, n_terms, mode_name) / trace_name)
            for row in rows:
                value = row.get(value_key)
                if value in {None, "", "None"}:
                    continue
                step = int(float(row["step"]))
                merged.setdefault(step, []).append(float(value))
        if not merged:
            continue
        xs = sorted(merged)
        ys = [sum(merged[x]) / len(merged[x]) for x in xs]
        ax.plot(xs, ys, label=mode_name, linewidth=1.5)
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
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_visualization_stage(config: Dict[str, Any]) -> None:
    paths = training_paths(config)
    stage_message(1, 3, "Writing resolved Pipeline II config snapshot")
    save_config(config, paths.config_used_path)

    stage_message(2, 3, "Collecting milestone summaries")
    rows = _collect_rows(config)
    _write_csv(paths.tables_root / "milestone_summary.csv", rows)
    write_json(paths.tables_root / "milestone_summary.json", rows)
    print(f"Saved milestone summary to: {paths.tables_root / 'milestone_summary.csv'}")

    stage_message(3, 3, "Writing Pipeline II figures")
    for experiment in get_experiments(config):
        n_terms = int(experiment["n_terms"])
        _plot_metric_for_terms(
            config=config,
            rows=rows,
            n_terms=n_terms,
            metric="step",
            ylabel="Steps to milestone",
            output_path=paths.figures_main_text_root / f"terms_{n_terms:02d}_steps_to_digit_milestone.png",
        )
        _plot_metric_for_terms(
            config=config,
            rows=rows,
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
        )
        _plot_trace(
            config=config,
            n_terms=n_terms,
            trace_name="train_trace.csv",
            value_key="true_mass",
            ylabel="True-sum mass",
            output_path=paths.figures_main_text_root / f"terms_{n_terms:02d}_true_mass_trace.png",
        )
        _plot_trace(
            config=config,
            n_terms=n_terms,
            trace_name="train_trace.csv",
            value_key="branch_count",
            ylabel="Branch count",
            output_path=paths.figures_appendix_root / f"terms_{n_terms:02d}_branch_count_trace.png",
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
