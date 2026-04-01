from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from statistics import mean, median
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt

from mnist_spll_common import ensure_dir, load_config, set_seed, stage_message
from mnist_spll_pipeline_core import (
    build_pipeline_context,
    build_stage_metadata,
    load_json,
    stage_config_snapshot,
    write_json,
)


def normalize_distribution(values: Sequence[float]) -> List[float]:
    total = float(sum(values))
    if total <= 0:
        return [0.0 for _ in values]
    return [float(v) / total for v in values]


def top_predictions(posterior: Sequence[float], k: int) -> List[Dict[str, float]]:
    indexed = sorted(enumerate(posterior), key=lambda item: item[1], reverse=True)[:k]
    return [{"sum": int(idx), "probability": float(prob)} for idx, prob in indexed]


def write_csv(path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["threshold_label"]].append(row)

    summary: List[Dict[str, Any]] = []
    for label, items in grouped.items():
        summary.append(
            {
                "threshold_label": label,
                "cutoff": items[0]["cutoff"],
                "experiments": len(items),
                "accuracy": sum(int(item["correct"]) for item in items) / len(items),
                "mean_runtime_sec": mean(item["runtime_sec"] for item in items),
                "median_runtime_sec": median(item["runtime_sec"] for item in items),
                "mean_confidence": mean(item["confidence"] for item in items),
                "mean_posterior_mass": mean(item["posterior_mass"] for item in items),
            }
        )
    summary.sort(key=lambda row: (row["cutoff"] is not None, float(row["cutoff"] or -1.0)))
    return summary


def save_plots(summary_rows: List[Dict[str, Any]], detailed_rows: List[Dict[str, Any]], term_counts: List[int], plots_dir) -> None:
    ensure_dir(plots_dir)

    if term_counts:
        plt.figure(figsize=(8, 4.5))
        plt.hist(term_counts, bins=range(min(term_counts), max(term_counts) + 2), align="left", rwidth=0.85)
        plt.xlabel("Number of digits summed")
        plt.ylabel("Experiment count")
        plt.title("Distribution of sampled term counts")
        plt.tight_layout()
        plt.savefig(plots_dir / "term_count_histogram.png", dpi=180)
        plt.close()

    if summary_rows:
        labels = [row["threshold_label"] for row in summary_rows]
        runtimes = [row["mean_runtime_sec"] for row in summary_rows]
        accuracies = [row["accuracy"] for row in summary_rows]

        plt.figure(figsize=(8, 4.5))
        plt.plot(labels, runtimes, marker="o")
        plt.xlabel("Inference setting")
        plt.ylabel("Mean runtime (s)")
        plt.title("Mean runtime by SPLL cutoff")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / "runtime_vs_cutoff.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 4.5))
        plt.plot(labels, accuracies, marker="o")
        plt.xlabel("Inference setting")
        plt.ylabel("Accuracy")
        plt.title("Prediction accuracy by SPLL cutoff")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / "accuracy_vs_cutoff.png", dpi=180)
        plt.close()

    if detailed_rows:
        grouped_term_runtime: Dict[int, List[float]] = defaultdict(list)
        for row in detailed_rows:
            grouped_term_runtime[int(row["n_terms"])].append(float(row["runtime_sec"]))
        term_keys = sorted(grouped_term_runtime)
        term_runtime_values = [mean(grouped_term_runtime[key]) for key in term_keys]
        plt.figure(figsize=(8, 4.5))
        plt.plot(term_keys, term_runtime_values, marker="o")
        plt.xlabel("Number of digits summed")
        plt.ylabel("Mean runtime (s)")
        plt.title("Mean runtime by term count")
        plt.tight_layout()
        plt.savefig(plots_dir / "runtime_vs_term_count.png", dpi=180)
        plt.close()


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


def run_visualization_stage(config: Dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    ctx = build_pipeline_context(config)

    stage_message(1, 3, "Loading raw staged experiments and inference runs")
    if not ctx.paths.inference_runs_path.exists():
        raise FileNotFoundError(
            f"Raw inference results not found at {ctx.paths.inference_runs_path}. Run the 'infer' step first."
        )
    experiments = load_payload_experiments(ctx.paths.staged_experiments_path)
    raw_runs = load_payload_runs(ctx.paths.inference_runs_path)

    stage_message(2, 3, "Computing derived metrics from raw posterior traces")
    top_n = int(ctx.inference_cfg.get("top_predictions_to_store", 5))
    detailed_rows: List[Dict[str, Any]] = []
    for run in raw_runs:
        posterior_raw = [float(value) for value in run["posterior_raw"]]
        posterior = normalize_distribution(posterior_raw)
        predicted_sum = int(max(range(len(posterior)), key=lambda idx: posterior[idx])) if posterior else 0
        confidence = float(posterior[predicted_sum]) if posterior else 0.0
        detailed_rows.append(
            {
                "experiment_id": int(run["experiment_id"]),
                "threshold_label": run["threshold_label"],
                "cutoff": run["cutoff"],
                "n_terms": int(run["n_terms"]),
                "true_sum": int(run["true_sum"]),
                "predicted_sum": predicted_sum,
                "correct": int(predicted_sum == int(run["true_sum"])),
                "runtime_sec": float(run["runtime_sec"]),
                "confidence": confidence,
                "posterior_mass": float(sum(posterior_raw)),
                "labels": str(run["labels"]),
                "global_indices": str(run["global_indices"]),
                "image_paths": str(run["image_paths"]),
                "top_predictions": str(top_predictions(posterior, top_n)),
            }
        )

    summary_rows = summarize_results(detailed_rows)

    stage_message(3, 3, "Writing tables and plots for visualization output")
    vis_root = ctx.paths.visualization_root
    plots_dir = ensure_dir(vis_root / "plots")
    write_csv(vis_root / "detailed_results.csv", detailed_rows)
    write_csv(vis_root / "summary_results.csv", summary_rows)
    write_json(
        vis_root / "summary_results.json",
        {
            "metadata": build_stage_metadata(
                config,
                "visualize_summary",
                extra={
                    "num_detailed_rows": len(detailed_rows),
                    "num_summary_rows": len(summary_rows),
                    "raw_inference_source": str(ctx.paths.inference_runs_path),
                },
            ),
            "summary": summary_rows,
        },
    )
    save_plots(
        summary_rows=summary_rows,
        detailed_rows=detailed_rows,
        term_counts=[int(exp["n_terms"]) for exp in experiments],
        plots_dir=plots_dir,
    )
    stage_config_snapshot(config, vis_root / "visualize_config_used.yaml")
    print(f"Saved visualization bundle to: {vis_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute summaries and plots from saved raw SPLL inference runs.")
    parser.add_argument("--config", required=True, help="Path to the shared YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_visualization_stage(config)


if __name__ == "__main__":
    main()
