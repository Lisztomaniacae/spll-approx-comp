from __future__ import annotations

from typing import Any, Dict

from pipeline2_analysis import (
    _collect_adaptive_topk_trace_rows,
    _collect_checkpoint_transfer_rows,
    _collect_milestone_aggregates,
    _collect_rows,
    _collect_run_summaries,
    _non_anchor_mode_names,
    _write_csv,
)
from pipeline2_config import get_checkpoint_transfer_config, get_experiments, get_inference_modes, training_paths
from pipeline2_plotting import (
    _plot_adaptive_topk_event_trace,
    _plot_adaptive_topk_search_iterations,
    _plot_checkpoint_transfer_metric_trajectory,
    _plot_combined_checkpoint_transfer_metric,
    _plot_dual_axis_checkpoint_bars,
    _plot_metric_for_terms,
    _plot_trace,
)
from pipeline_support import run_configured_stage_cli, save_config, stage_message, write_json


def run_visualization_stage(config: Dict[str, Any]) -> None:
    paths = training_paths(config)
    paths.ensure_visualization_dirs()
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
        _plot_dual_axis_checkpoint_bars(
            config=config,
            rows=rows,
            run_summaries=run_summaries,
            n_terms=n_terms,
            output_path=paths.figures_main_text_root / f"terms_{n_terms:02d}_posterior_checkpoint_progress_dual_axis.png",
        )
        _plot_metric_for_terms(
            config=config,
            rows=rows,
            run_summaries=run_summaries,
            n_terms=n_terms,
            metric="step",
            ylabel="Steps to posterior checkpoint",
            output_path=paths.figures_appendix_root / f"terms_{n_terms:02d}_steps_to_true_sum_posterior_checkpoint.png",
        )
        _plot_metric_for_terms(
            config=config,
            rows=rows,
            run_summaries=run_summaries,
            n_terms=n_terms,
            metric="elapsed_seconds",
            ylabel="Seconds to posterior checkpoint",
            output_path=paths.figures_appendix_root / f"terms_{n_terms:02d}_time_to_true_sum_posterior_checkpoint.png",
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
            legend_loc="upper right",
            legend_bbox=None,
            show_footer=False,
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
            legend_loc="upper left",
            legend_bbox=None,
            show_footer=False,
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
        checkpoint_transfer_cfg = get_checkpoint_transfer_config(config)
        anchor_mode_name = str(checkpoint_transfer_cfg.get("anchor_mode_name", "exact"))
        requested_transfer_modes = checkpoint_transfer_cfg.get("mode_names")
        combined_transfer_modes = _non_anchor_mode_names(config, anchor_mode_name)
        if isinstance(requested_transfer_modes, list):
            allowed_transfer_modes = {str(v) for v in requested_transfer_modes}
            combined_transfer_modes = [mode_name for mode_name in combined_transfer_modes if mode_name in allowed_transfer_modes]
        _plot_combined_checkpoint_transfer_metric(
            config=config,
            n_terms=n_terms,
            mode_names=combined_transfer_modes,
            anchor_mode_name=anchor_mode_name,
            value_key="loss",
            ylabel="Training loss",
            output_path=paths.figures_main_text_root / f"terms_{n_terms:02d}_loss_exact_vs_approx_combined.png",
            smooth_window=smooth_window,
        )
        _plot_combined_checkpoint_transfer_metric(
            config=config,
            n_terms=n_terms,
            mode_names=combined_transfer_modes,
            anchor_mode_name=anchor_mode_name,
            value_key="true_mass",
            ylabel="True-sum posterior (%)",
            output_path=paths.figures_main_text_root / f"terms_{n_terms:02d}_true_mass_exact_vs_approx_combined.png",
            smooth_window=smooth_window,
            as_percent=True,
        )
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
                output_path=paths.figures_appendix_root / f"terms_{n_terms:02d}_loss_exact_vs_{mode_name}.png",
                smooth_window=smooth_window,
            )
            _plot_checkpoint_transfer_metric_trajectory(
                config=config,
                n_terms=n_terms,
                mode_name=mode_name,
                anchor_mode_name=anchor_mode_name,
                value_key="true_mass",
                ylabel="True-sum posterior (%)",
                output_path=paths.figures_appendix_root / f"terms_{n_terms:02d}_true_mass_exact_vs_{mode_name}.png",
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
    run_configured_stage_cli(
        run_visualization_stage,
        description="Visualize Pipeline II SPLL training results.",
        config_help="Path to the Pipeline II YAML config.",
    )


if __name__ == "__main__":
    main()
