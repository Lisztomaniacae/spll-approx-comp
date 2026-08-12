from __future__ import annotations

from typing import Any, Dict

from pipeline1_analysis import (
    adaptive_topk_search_rows,
    add_exact_baseline_columns,
    add_paired_accuracy_delta_intervals,
    load_payload_experiments,
    load_payload_runs,
    ordered_threshold_labels,
    prepare_detailed_rows,
    pretty_threshold_label,
    summarize_groups,
    set_analysis_seed,
    write_csv,
)
from pipeline1_config import build_pipeline_context, GLOBAL_CUTOFF_MODE
from pipeline1_plotting import (
    heatmap_specs,
    plot_accuracy_delta_vs_cutoff,
    plot_adaptive_topk_search_iterations,
    plot_heatmap_metric,
    plot_mnist_lookup_accuracy_tradeoff,
    plot_overhead_exact_vs_zero,
    plot_pareto_tradeoff,
    plot_runtime_vs_cutoff,
    plot_target_vs_achieved,
    plot_tradeoff_matrix_by_terms,
    plot_true_candidate_metric_vs_cutoff,
    write_bundle_readme,
    write_tradeoff_alternatives_readme,
)
from pipeline_support import (
    build_stage_metadata,
    ensure_dir,
    run_configured_stage_cli,
    save_config,
    stage_message,
    write_json,
)


def run_visualization_stage(config: Dict[str, Any]) -> None:
    set_analysis_seed(int(config.get("seed", 42)))

    ctx = build_pipeline_context(config)
    threshold_order = ordered_threshold_labels(config)
    cutoff_mode = GLOBAL_CUTOFF_MODE

    stage_message(1, 3, "Loading raw staged experiments and inference runs")
    if not ctx.paths.inference_runs_path.exists():
        raise FileNotFoundError(
            f"Raw inference results not found at {ctx.paths.inference_runs_path}. Run the 'infer' step first."
        )
    experiments = load_payload_experiments(ctx.paths.staged_experiments_path)
    raw_runs = load_payload_runs(ctx.paths.inference_runs_path)

    stage_message(2, 3, "Computing derived metrics and exact-baseline deltas")
    top_n = int(ctx.inference_cfg.get("top_predictions_to_store", 5))
    detailed_rows = prepare_detailed_rows(raw_runs, top_n=top_n)
    adaptive_search_rows = adaptive_topk_search_rows(raw_runs)
    summary_by_terms = summarize_groups(
        detailed_rows,
        group_keys=["cutoff_mode", "model_id", "n_terms", "threshold_label", "cutoff"],
        threshold_order=threshold_order,
    )
    add_exact_baseline_columns(summary_by_terms, ["cutoff_mode", "model_id", "n_terms", "threshold_label", "cutoff"])
    add_paired_accuracy_delta_intervals(
        summary_by_terms,
        detailed_rows,
        ["cutoff_mode", "model_id", "n_terms", "threshold_label", "cutoff"],
        bootstrap_samples=2000,
        seed=int(config.get("seed", 42)),
    )

    mode_rows = [row for row in summary_by_terms if str(row.get("cutoff_mode", "global")) == cutoff_mode]
    if not mode_rows:
        raise ValueError(f"No summary rows found for cutoff mode: {cutoff_mode}")

    standard_rows = [
        row for row in mode_rows if str(row.get("visualization_group", "main")) == "main"
    ]
    biased_tradeoff_rows = [
        row for row in mode_rows if str(row.get("visualization_group", "main")) == "biased_tradeoff"
    ]
    if not standard_rows:
        raise ValueError("No standard model rows found for visualization_group='main'.")
    standard_model_ids = {str(row["model_id"]) for row in standard_rows}

    stage_message(3, 3, "Writing tables and figure bundle")
    vis_root = ctx.paths.visualization_root
    table_dir = ensure_dir(vis_root / "tables")
    figure_root = ensure_dir(vis_root / "figures")
    main_dir = ensure_dir(figure_root / "main_text")
    appendix_dir = ensure_dir(figure_root / "appendix")
    heatmap_dir = ensure_dir(appendix_dir / "heatmaps")
    tradeoff_alt_dir = ensure_dir(figure_root / "tradeoff_alternatives")

    write_csv(table_dir / "detailed_results.csv", detailed_rows)
    write_csv(table_dir / "summary_results.csv", summary_by_terms)
    write_csv(table_dir / "adaptive_topk_search_trace.csv", adaptive_search_rows)
    write_json(table_dir / "adaptive_topk_search_trace.json", adaptive_search_rows)
    write_json(
        table_dir / "summary_results.json",
        {
            "metadata": build_stage_metadata(
                config,
                "visualize",
                extra={
                    "num_detailed_rows": len(detailed_rows),
                    "num_summary_rows": len(summary_by_terms),
                    "num_adaptive_topk_search_rows": len(adaptive_search_rows),
                    "raw_inference_source": str(ctx.paths.inference_runs_path),
                    "cutoff_mode": cutoff_mode,
                },
            ),
            "summary_by_terms": summary_by_terms,
        },
    )

    term_counts = sorted({int(exp["n_terms"]) for exp in experiments})

    plot_pareto_tradeoff(
        summary_rows=standard_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "runtime_accuracy_tradeoff_by_terms.png",
    )
    plot_mnist_lookup_accuracy_tradeoff(
        summary_rows=standard_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "mnist_lookup_accuracy_tradeoff_by_terms.png",
    )
    if biased_tradeoff_rows:
        plot_pareto_tradeoff(
            summary_rows=biased_tradeoff_rows,
            term_counts=term_counts,
            threshold_order=threshold_order,
            output_path=main_dir / "runtime_accuracy_tradeoff_biased_models_by_terms.png",
            title="Runtime–accuracy tradeoff relative to exact inference — biased models",
        )
    plot_tradeoff_matrix_by_terms(
        summary_rows=standard_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=tradeoff_alt_dir / "01_tradeoff_matrix_by_terms.png",
    )
    write_tradeoff_alternatives_readme(tradeoff_alt_dir / "README.txt")
    plot_runtime_vs_cutoff(
        summary_rows=standard_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "runtime_vs_cutoff_by_terms.png",
    )
    plot_accuracy_delta_vs_cutoff(
        summary_rows=standard_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "accuracy_delta_vs_exact_by_terms.png",
    )
    plot_true_candidate_metric_vs_cutoff(
        summary_rows=standard_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "true_candidate_runtime_vs_cutoff_by_terms.png",
        metric_key="median_true_candidate_runtime_sec",
        ylabel="Median true-sum-only runtime (s)",
        title="True-sum-only runtime vs pruning threshold",
        yscale="log",
    )
    plot_true_candidate_metric_vs_cutoff(
        summary_rows=standard_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "true_candidate_branch_count_vs_cutoff_by_terms.png",
        metric_key="mean_true_candidate_branch_count",
        ylabel="Mean true-sum branch count",
        title="True-sum branch count vs pruning threshold",
    )
    plot_true_candidate_metric_vs_cutoff(
        summary_rows=standard_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "true_candidate_survival_vs_cutoff_by_terms.png",
        metric_key="true_candidate_survival_rate",
        ylabel="True-sum survival rate",
        title="True-sum survival rate vs pruning threshold",
        ylim=(0.0, 1.05),
    )
    plot_true_candidate_metric_vs_cutoff(
        summary_rows=standard_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "true_candidate_probability_vs_cutoff_by_terms.png",
        metric_key="mean_true_candidate_normalized_probability",
        ylabel="Mean normalized true-sum probability",
        title="Probability assigned to the true sum vs pruning threshold",
        ylim=(0.0, 1.05),
    )

    overhead_rows = plot_overhead_exact_vs_zero(
        summary_rows=standard_rows,
        term_counts=term_counts,
        threshold_order=threshold_order,
        output_path=main_dir / "overhead_exact_vs_zero_cutoff_by_terms.png",
    )
    write_csv(table_dir / "overhead_exact_vs_zero_summary.csv", overhead_rows)

    target_rows = plot_target_vs_achieved(
        summary_rows=standard_rows,
        output_path=appendix_dir / "target_vs_achieved_model_accuracy.png",
    )
    write_csv(table_dir / "model_accuracy_targets.csv", target_rows)
    plot_adaptive_topk_search_iterations(
        search_rows=[
            row for row in adaptive_search_rows
            if str(row.get("cutoff_mode", "global")) == cutoff_mode
            and str(row.get("model_id")) in standard_model_ids
        ],
        summary_rows=standard_rows,
        term_counts=term_counts,
        value_key="candidate_cutoff",
        ylabel="Candidate TOP_K_CUTOFF",
        output_path=appendix_dir / "adaptive_topk_search_cutoff_iterations_by_terms.png",
    )
    plot_adaptive_topk_search_iterations(
        search_rows=[
            row for row in adaptive_search_rows
            if str(row.get("cutoff_mode", "global")) == cutoff_mode
            and str(row.get("model_id")) in standard_model_ids
        ],
        summary_rows=standard_rows,
        term_counts=term_counts,
        value_key="mean_surviving_mass",
        ylabel="Mean surviving posterior mass",
        output_path=appendix_dir / "adaptive_topk_search_mass_iterations_by_terms.png",
        show_target=True,
        ylim=(0.0, 1.05),
    )

    for spec in heatmap_specs():
        plot_heatmap_metric(
            summary_rows=standard_rows,
            spec=spec,
            term_counts=term_counts,
            threshold_order=threshold_order,
            output_path=heatmap_dir / spec.filename,
        )

    root_readme_lines = [
        "Visualization bundle generated by visualize_results.py",
        "",
        "Cutoff mode used: global",
        f"Terms shown: {', '.join(str(value) for value in term_counts)}",
        f"Cutoffs shown: {', '.join(pretty_threshold_label(label) for label in threshold_order)}",
        "",
        "Figures live under figures/...",
        "Tables live under tables/...",
    ]

    write_bundle_readme(
        figure_root / "README.txt",
        term_counts,
        threshold_order,
        include_biased_tradeoff=bool(biased_tradeoff_rows),
    )
    (vis_root / "README.txt").write_text("\n".join(root_readme_lines), encoding="utf-8")
    save_config(config, vis_root / "visualize_config_used.yaml")
    print(f"Saved visualization bundle to: {vis_root}")


def main() -> None:
    run_configured_stage_cli(
        run_visualization_stage,
        description="Compute tables and figure bundles from saved raw SPLL inference runs.",
        config_help="Path to the shared YAML config.",
    )


if __name__ == "__main__":
    main()
