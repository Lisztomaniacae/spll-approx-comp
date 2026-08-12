from __future__ import annotations

import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from mnist_model import resolve_device, set_seed
from pipeline2_adaptive import get_runtime_top_k_cutoff, tune_adaptive_top_k_cutoff
from pipeline2_artifacts import assert_compiled_artifacts_exist, load_compiled_training_module
from pipeline2_checkpoint_transfer import (
    _build_aggregate_exact_checkpoints,
    _checkpointing_cfg,
    _materialize_aggregate_exact_anchor_checkpoints,
    _posterior_checkpoint_path,
    _run_checkpoint_transfer_stage,
    _threshold_key,
    _write_posterior_checkpoints_json,
)
from pipeline2_config import (
    get_checkpoint_transfer_config,
    compiled_program_path,
    get_experiments,
    get_inference_modes,
    get_seeds,
    initial_checkpoint_path,
    run_dir,
    schedule_manifest_path,
    training_paths,
)
from pipeline2_data import generate_sum_case, load_mnist_train_dataset
from pipeline2_runtime import (
    DifferentiableReadMNIST,
    TensorLRUCache,
    cleanup_torch,
    load_initial_model,
    make_optimizer,
    open_csv_trace,
    recent_mean,
    run_preflight,
    save_training_checkpoint,
    train_sum_batch,
    write_failure_report,
)
from pipeline_support import (
    TerminalProgressBar,
    ensure_dir,
    load_json,
    run_configured_stage_cli,
    save_config,
    stage_message,
    write_json,
)


def _write_milestones_json(path: Path, milestones_state: Dict[str, Dict[str, Any]]) -> None:
    write_json(path, {"milestones": milestones_state})


def _rotated_modes(modes: List[Dict[str, Any]], offset: int) -> List[Dict[str, Any]]:
    if not modes:
        return []
    shift = int(offset) % len(modes)
    return modes[shift:] + modes[:shift]


def _train_one_run(
    *,
    config: Dict[str, Any],
    split_manifest: Dict[str, Any],
    dataset: Any,
    seed: int,
    experiment: Dict[str, Any],
    mode: Dict[str, Any],
    mode_order_position: int = 0,
    mode_order_offset: int = 0,
    run_order_index: Optional[int] = None,
) -> None:
    paths = training_paths(config)
    n_terms = int(experiment["n_terms"])
    max_steps = int(experiment["max_steps"])
    mode_name = str(mode["name"])
    this_run_dir = run_dir(paths, seed, n_terms, mode_name)
    ensure_dir(this_run_dir)
    ensure_dir(this_run_dir / "checkpoints")

    context = {
        "seed": seed,
        "n_terms": n_terms,
        "mode": mode_name,
        "artifact_name": mode.get("artifact_name", mode_name),
        "top_k_cutoff": mode.get("top_k_cutoff"),
        "adaptive_top_k": bool(mode.get("adaptive_top_k", False)),
        "posterior_mass_target": mode.get("posterior_mass_target"),
        "cutoff_search": mode.get("cutoff_search"),
        "mode_order_position": int(mode_order_position),
        "mode_order_offset": int(mode_order_offset),
        "run_order_index": None if run_order_index is None else int(run_order_index),
    }
    csv_stack = ExitStack()
    cache: Optional[TensorLRUCache] = None
    try:
        set_seed(seed)
        train_cfg = config.get("training", {})
        device = resolve_device(str(train_cfg.get("device", "auto")), bool(train_cfg.get("require_mps", False)))
        model = load_initial_model(config, initial_checkpoint_path(paths, seed, n_terms), device)
        optimizer = make_optimizer(config, model)

        module = load_compiled_training_module(compiled_program_path(paths, n_terms, mode), n_terms, mode_name)
        data_cfg = config.get("data", {})
        cache = TensorLRUCache(
            dataset,
            device=device,
            cache_device=str(data_cfg.get("image_cache_device", "device")),
            max_items=int(data_cfg.get("image_cache_max_items", 4096)),
            strategy=str(data_cfg.get("image_cache_strategy", "lru")),
        )
        read_mnist = DifferentiableReadMNIST(model, cache, device)
        setattr(module, "readMNist", read_mnist)
        adaptive_cutoff_state: Optional[Dict[str, Any]] = tune_adaptive_top_k_cutoff(
            config=config,
            mode=mode,
            module=module,
            model=model,
            cache=cache,
            device=device,
            split_manifest=split_manifest,
            seed=seed,
            n_terms=n_terms,
            step=0,
            reason="preflight",
        )
        setattr(module, "readMNist", read_mnist)

        loss_epsilon = float(train_cfg.get("loss_epsilon", 1.0e-12))
        sum_batch_size = max(1, int(train_cfg.get("sum_batch_size", 1)))
        run_preflight(
            module=module,
            model=model,
            optimizer=optimizer,
            split_manifest=split_manifest,
            seed=seed,
            n_terms=n_terms,
            loss_epsilon=loss_epsilon,
        )

        checkpoint_cfg = _checkpointing_cfg(config)
        checkpoint_thresholds = list(checkpoint_cfg["posterior_thresholds"])
        checkpoint_window = int(checkpoint_cfg["rolling_window_updates"])
        checkpointing_enabled = bool(checkpoint_cfg["enabled"])
        train_handle, train_writer = csv_stack.enter_context(open_csv_trace(
            this_run_dir / "train_trace.csv",
            [
                "step",
                "optimizer_update",
                "batch_size",
                "case_start_step",
                "case_end_step",
                "elapsed_seconds",
                "loss",
                "true_mass",
                "zero_true_mass",
                "loss_recent_mean",
                "true_mass_recent_mean",
                "zero_true_mass_recent_rate",
                "branch_count",
                "branch_count_mean",
                "branch_count_total",
                "read_mnist_lookup_mean",
                "read_mnist_lookup_total",
                "read_mnist_lookup_cumulative",
                "grad_norm",
                "top_k_cutoff_runtime",
                "posterior_mass_target",
                "cutoff_search_mean_surviving_posterior_mass",
                "cutoff_search_abs_error",
            ],
        ))
        topk_event_handle, topk_event_writer = csv_stack.enter_context(open_csv_trace(
            this_run_dir / "adaptive_topk_events.csv",
            [
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
        ))
        topk_search_handle, topk_search_writer = csv_stack.enter_context(open_csv_trace(
            this_run_dir / "adaptive_topk_search_trace.csv",
            [
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
        ))

        posterior_checkpoints_state: Dict[str, Dict[str, Any]] = {
            _threshold_key(threshold): {
                "reached": False,
                "step": None,
                "elapsed_seconds": None,
                "rolling_true_mass": None,
                "rolling_loss": None,
                "threshold": float(threshold),
                "metric": "true_sum_posterior_recent_mean",
                "checkpoint_path": None,
                "cumulative_read_mnist_lookup_calls": None,
            }
            for threshold in checkpoint_thresholds
        }
        # Backward-compatible alias for older visualization/table code.  The value
        # is now a true-sum posterior training checkpoint, not a validation accuracy milestone.
        milestones_state: Dict[str, Dict[str, Any]] = {
            key: {
                "reached": False,
                "step": None,
                "elapsed_seconds": None,
                "digit_accuracy": None,
                "sum_posterior_accuracy": None,
                "validation_metric": "true_sum_posterior_recent_mean",
                "cumulative_read_mnist_lookup_calls": None,
            }
            for key in posterior_checkpoints_state
        }

        recent_losses: Deque[float] = deque(maxlen=max(1, checkpoint_window))
        recent_masses: Deque[float] = deque(maxlen=max(1, checkpoint_window))
        recent_zeros: Deque[float] = deque(maxlen=max(1, checkpoint_window))
        started_at = time.perf_counter()

        def write_adaptive_topk_trace(step: int, reason: str, cutoff_state: Dict[str, Any]) -> None:
            evaluations = list(cutoff_state.get("evaluations") or [])
            topk_event_writer.writerow(
                {
                    "step": int(step),
                    "reason": str(reason),
                    "runtime_top_k_cutoff": cutoff_state.get("runtime_top_k_cutoff"),
                    "posterior_mass_target": cutoff_state.get("posterior_mass_target"),
                    "mean_surviving_posterior_mass": cutoff_state.get("mean_surviving_posterior_mass"),
                    "abs_error": cutoff_state.get("abs_error"),
                    "iterations": cutoff_state.get("iterations"),
                    "converged": cutoff_state.get("converged"),
                    "probe_cases": cutoff_state.get("probe_cases"),
                    "search_runtime_sec": cutoff_state.get("search_runtime_sec"),
                    "evaluation_count": len(evaluations),
                }
            )
            for evaluation_index, evaluation in enumerate(evaluations):
                topk_search_writer.writerow(
                    {
                        "step": int(step),
                        "reason": str(reason),
                        "evaluation_index": int(evaluation_index),
                        "candidate_cutoff": evaluation.get("cutoff"),
                        "mean_surviving_mass": evaluation.get("mean_surviving_mass"),
                        "posterior_mass_target": cutoff_state.get("posterior_mass_target"),
                        "runtime_top_k_cutoff": cutoff_state.get("runtime_top_k_cutoff"),
                        "selected_mean_surviving_posterior_mass": cutoff_state.get("mean_surviving_posterior_mass"),
                        "selected_abs_error": cutoff_state.get("abs_error"),
                        "iterations": cutoff_state.get("iterations"),
                        "converged": cutoff_state.get("converged"),
                    }
                )
            topk_event_handle.flush()
            topk_search_handle.flush()

        if adaptive_cutoff_state is not None:
            write_adaptive_topk_trace(0, "preflight", adaptive_cutoff_state)

        def refresh_adaptive_cutoff(step: int, reason: str) -> Optional[Dict[str, Any]]:
            nonlocal adaptive_cutoff_state
            cutoff_state = tune_adaptive_top_k_cutoff(
                config=config,
                mode=mode,
                module=module,
                model=model,
                cache=cache,
                device=device,
                split_manifest=split_manifest,
                seed=seed,
                n_terms=n_terms,
                step=step,
                reason=reason,
            )
            setattr(module, "readMNist", read_mnist)
            if cutoff_state is not None:
                adaptive_cutoff_state = cutoff_state
                write_adaptive_topk_trace(step, reason, cutoff_state)
            return cutoff_state

        progress = TerminalProgressBar(
            max_steps,
            desc=f"Train terms={n_terms} mode={mode_name} seed={seed}",
            unit="iterations",
            enabled=bool(config.get("show_progress", True)),
        )
        cases_seen = 0
        optimizer_update = 0
        cumulative_read_mnist_lookup_calls = 0
        last_loss: Optional[float] = None
        last_true_mass: Optional[float] = None
        last_zero_rate: Optional[float] = None
        last_grad_norm: Optional[float] = None
        while cases_seen < max_steps:
            case_start_step = cases_seen + 1
            case_end_step = min(max_steps, cases_seen + sum_batch_size)
            batch_cases = [
                generate_sum_case(
                    split_manifest,
                    base_seed=seed,
                    n_terms=n_terms,
                    step=case_step,
                    split="train",
                )
                for case_step in range(case_start_step, case_end_step + 1)
            ]
            optimizer_update += 1
            batch_stats = train_sum_batch(
                module=module,
                model=model,
                optimizer=optimizer,
                cache=cache,
                device=device,
                cases=batch_cases,
                loss_epsilon=loss_epsilon,
            )
            cases_seen = case_end_step

            elapsed = time.perf_counter() - started_at
            last_loss = float(batch_stats["loss"])
            last_true_mass = float(batch_stats["true_mass"])
            last_zero_rate = float(batch_stats["zero_true_mass"])
            last_grad_norm = float(batch_stats["grad_norm"])
            recent_losses.append(last_loss)
            recent_masses.append(last_true_mass)
            recent_zeros.append(float(last_zero_rate or 0.0))
            rolling_loss = recent_mean(list(recent_losses))
            rolling_true_mass = recent_mean(list(recent_masses))
            rolling_zero_rate = recent_mean(list(recent_zeros))
            branch_count_mean = batch_stats["branch_count_mean"]
            branch_count_total = batch_stats["branch_count_total"]
            read_mnist_lookup_mean = batch_stats["read_mnist_lookup_mean"]
            read_mnist_lookup_total = int(batch_stats["read_mnist_lookup_total"] or 0)
            cumulative_read_mnist_lookup_calls += read_mnist_lookup_total
            train_writer.writerow(
                {
                    "step": cases_seen,
                    "optimizer_update": optimizer_update,
                    "batch_size": batch_stats["batch_size"],
                    "case_start_step": case_start_step,
                    "case_end_step": case_end_step,
                    "elapsed_seconds": elapsed,
                    "loss": last_loss,
                    "true_mass": last_true_mass,
                    "zero_true_mass": last_zero_rate,
                    "loss_recent_mean": rolling_loss,
                    "true_mass_recent_mean": rolling_true_mass,
                    "zero_true_mass_recent_rate": rolling_zero_rate,
                    "branch_count": branch_count_mean,
                    "branch_count_mean": branch_count_mean,
                    "branch_count_total": branch_count_total,
                    "read_mnist_lookup_mean": read_mnist_lookup_mean,
                    "read_mnist_lookup_total": read_mnist_lookup_total,
                    "read_mnist_lookup_cumulative": cumulative_read_mnist_lookup_calls,
                    "grad_norm": last_grad_norm,
                    "top_k_cutoff_runtime": get_runtime_top_k_cutoff(module, mode),
                    "posterior_mass_target": mode.get("posterior_mass_target"),
                    "cutoff_search_mean_surviving_posterior_mass": (adaptive_cutoff_state or {}).get("mean_surviving_posterior_mass"),
                    "cutoff_search_abs_error": (adaptive_cutoff_state or {}).get("abs_error"),
                }
            )


            if checkpointing_enabled and rolling_true_mass is not None:
                reached_new = False
                for threshold in checkpoint_thresholds:
                    key = _threshold_key(threshold)
                    if not posterior_checkpoints_state[key]["reached"] and float(rolling_true_mass) >= float(threshold):
                        checkpoint_path = _posterior_checkpoint_path(this_run_dir, threshold)
                        posterior_checkpoints_state[key] = {
                            "reached": True,
                            "step": int(cases_seen),
                            "elapsed_seconds": float(elapsed),
                            "rolling_true_mass": float(rolling_true_mass),
                            "rolling_loss": None if rolling_loss is None else float(rolling_loss),
                            "threshold": float(threshold),
                            "metric": "true_sum_posterior_recent_mean",
                            "checkpoint_path": str(checkpoint_path),
                            "cumulative_read_mnist_lookup_calls": int(cumulative_read_mnist_lookup_calls),
                        }
                        milestones_state[key] = {
                            "reached": True,
                            "step": int(cases_seen),
                            "elapsed_seconds": float(elapsed),
                            "digit_accuracy": float(rolling_true_mass),
                            "sum_posterior_accuracy": float(rolling_true_mass),
                            "validation_metric": "true_sum_posterior_recent_mean",
                            "cumulative_read_mnist_lookup_calls": int(cumulative_read_mnist_lookup_calls),
                        }
                        save_training_checkpoint(
                            checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            step=cases_seen,
                            elapsed_seconds=elapsed,
                            digit_accuracy_value=float(rolling_true_mass),
                            config=config,
                            seed=seed,
                            n_terms=n_terms,
                            mode=mode,
                            validation_metric_name="true_sum_posterior_rolling_mean",
                            sum_posterior_accuracy=float(rolling_true_mass),
                        )
                        reached_new = True
                if reached_new:
                    _write_posterior_checkpoints_json(this_run_dir / "posterior_checkpoints.json", posterior_checkpoints_state)
                    _write_milestones_json(this_run_dir / "milestones.json", milestones_state)

            if optimizer_update % 25 == 0 or cases_seen >= max_steps:
                train_handle.flush()
            progress.update(batch_stats["batch_size"], postfix=f"loss={last_loss:.4g}, p={last_true_mass:.4g}")

        progress.finish(postfix="max_steps reached")

        final_elapsed = time.perf_counter() - started_at
        # Retune adaptive modes at the end so summaries record the final runtime cutoff,
        # but do not run held-out validation as part of the redesigned benchmark.
        refresh_adaptive_cutoff(cases_seen, "final")
        final_rolling_loss = recent_mean(list(recent_losses))
        final_rolling_true_mass = recent_mean(list(recent_masses))
        final_rolling_zero = recent_mean(list(recent_zeros))
        save_training_checkpoint(
            this_run_dir / "checkpoints" / "final.pt",
            model=model,
            optimizer=optimizer,
            step=cases_seen,
            elapsed_seconds=final_elapsed,
            digit_accuracy_value=float(final_rolling_true_mass if final_rolling_true_mass is not None else (last_true_mass or 0.0)),
            config=config,
            seed=seed,
            n_terms=n_terms,
            mode=mode,
            validation_metric_name="true_sum_posterior_rolling_mean",
            sum_posterior_accuracy=float(final_rolling_true_mass if final_rolling_true_mass is not None else (last_true_mass or 0.0)),
        )
        _write_posterior_checkpoints_json(this_run_dir / "posterior_checkpoints.json", posterior_checkpoints_state)
        _write_milestones_json(this_run_dir / "milestones.json", milestones_state)
        write_json(
            this_run_dir / "run_summary.json",
            {
                "seed": int(seed),
                "n_terms": n_terms,
                "mode_name": mode_name,
                "artifact_name": mode.get("artifact_name", mode_name),
                "top_k_cutoff": mode.get("top_k_cutoff"),
                "adaptive_top_k": bool(mode.get("adaptive_top_k", False)),
                "posterior_mass_target": mode.get("posterior_mass_target"),
                "cutoff_search": mode.get("cutoff_search"),
                "runtime_top_k_cutoff": get_runtime_top_k_cutoff(module, mode),
                "last_cutoff_search": adaptive_cutoff_state,
                "mode_order_position": int(mode_order_position),
                "mode_order_offset": int(mode_order_offset),
                "run_order_index": None if run_order_index is None else int(run_order_index),
                "max_steps": max_steps,
                "sum_batch_size": sum_batch_size,
                "completed_training_cases": cases_seen,
                "optimizer_updates": optimizer_update,
                "checkpointing": checkpoint_cfg,
                "posterior_checkpoints": posterior_checkpoints_state,
                "validation_enabled_for_training": False,
                "final_elapsed_seconds": final_elapsed,
                "final_loss": last_loss,
                "final_true_mass": last_true_mass,
                "final_zero_true_mass": last_zero_rate,
                "final_loss_recent_mean": final_rolling_loss,
                "final_true_mass_recent_mean": final_rolling_true_mass,
                "final_zero_true_mass_recent_rate": final_rolling_zero,
                "final_digit_accuracy": None,
                "final_sum_posterior_accuracy": None,
                "validation_metric": "not_used",
                "final_cumulative_read_mnist_lookup_calls": int(cumulative_read_mnist_lookup_calls),
                "milestones": milestones_state,
            },
        )
    except BaseException as exc:
        write_failure_report(this_run_dir / "failure_report.json", error=exc, context=context)
        raise
    finally:
        csv_stack.close()
        if cache is not None:
            cache.clear()
        cleanup_torch()


def run_train_stage(config: Dict[str, Any]) -> None:
    paths = training_paths(config)
    paths.ensure_training_dirs()
    stage_message(1, 5, "Writing resolved Pipeline II config snapshot")
    save_config(config, paths.config_used_path)

    stage_message(2, 5, "Verifying prepared manifests, checkpoints, and compiled SPLL artifacts")
    if not paths.split_manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {paths.split_manifest_path}. Run prepare first.")
    split_manifest = load_json(paths.split_manifest_path)
    for seed in get_seeds(config):
        for experiment in get_experiments(config):
            n_terms = int(experiment["n_terms"])
            if not initial_checkpoint_path(paths, seed, n_terms).exists():
                raise FileNotFoundError(f"Missing initial checkpoint: {initial_checkpoint_path(paths, seed, n_terms)}. Run prepare first.")
            if not schedule_manifest_path(paths, seed, n_terms).exists():
                raise FileNotFoundError(f"Missing schedule manifest: {schedule_manifest_path(paths, seed, n_terms)}. Run prepare first.")
    assert_compiled_artifacts_exist(config, paths)

    stage_message(3, 5, "Loading MNIST train partition")
    dataset = load_mnist_train_dataset(config)

    stage_message(4, 5, "Training through generated SPLL artifacts")
    seeds = get_seeds(config)
    experiments = get_experiments(config)
    inference_modes = get_inference_modes(config)
    total = len(seeds) * len(experiments) * len(inference_modes)
    outer = TerminalProgressBar(total, desc="Pipeline II runs", unit="runs", enabled=bool(config.get("show_progress", True)))
    run_order_index = 0
    for seed_index, seed in enumerate(seeds):
        for experiment_index, experiment in enumerate(experiments):
            mode_order_offset = seed_index + experiment_index
            for mode_order_position, mode in enumerate(_rotated_modes(inference_modes, mode_order_offset)):
                _train_one_run(
                    config=config,
                    split_manifest=split_manifest,
                    dataset=dataset,
                    seed=seed,
                    experiment=experiment,
                    mode=mode,
                    mode_order_position=mode_order_position,
                    mode_order_offset=mode_order_offset,
                    run_order_index=run_order_index,
                )
                outer.update(postfix=f"seed={seed}, terms={experiment['n_terms']}, mode={mode['name']}")
                run_order_index += 1
    outer.finish(postfix="done")

    stage_message(5, 5, "Computing aggregate exact checkpoints and running checkpoint-transfer approximations")
    transfer_cfg = get_checkpoint_transfer_config(config)
    anchor_mode_name = str(transfer_cfg.get("anchor_mode_name", "exact"))
    for experiment in experiments:
        n_terms = int(experiment["n_terms"])
        _build_aggregate_exact_checkpoints(
            config,
            n_terms=n_terms,
            anchor_mode_name=anchor_mode_name,
            include_final_checkpoint=bool(transfer_cfg.get("include_final_checkpoint", True)),
        )
        _materialize_aggregate_exact_anchor_checkpoints(
            config,
            split_manifest,
            dataset,
            n_terms=n_terms,
            anchor_mode_name=anchor_mode_name,
        )
    _run_checkpoint_transfer_stage(config, split_manifest, dataset)


def run_checkpoint_transfer_only_stage(config: Dict[str, Any]) -> None:
    paths = training_paths(config)
    paths.ensure_training_dirs()
    stage_message(1, 4, "Writing resolved Pipeline II config snapshot")
    save_config(config, paths.config_used_path)

    stage_message(2, 4, "Verifying prepared manifests and compiled SPLL artifacts")
    if not paths.split_manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {paths.split_manifest_path}. Run prepare first.")
    split_manifest = load_json(paths.split_manifest_path)
    assert_compiled_artifacts_exist(config, paths)

    stage_message(3, 4, "Loading MNIST train partition")
    dataset = load_mnist_train_dataset(config)

    stage_message(4, 4, "Computing aggregate exact checkpoints and running checkpoint-transfer approximations")
    transfer_cfg = get_checkpoint_transfer_config(config)
    anchor_mode_name = str(transfer_cfg.get("anchor_mode_name", "exact"))
    for experiment in get_experiments(config):
        n_terms = int(experiment["n_terms"])
        _build_aggregate_exact_checkpoints(
            config,
            n_terms=n_terms,
            anchor_mode_name=anchor_mode_name,
            include_final_checkpoint=bool(transfer_cfg.get("include_final_checkpoint", True)),
        )
        _materialize_aggregate_exact_anchor_checkpoints(
            config,
            split_manifest,
            dataset,
            n_terms=n_terms,
            anchor_mode_name=anchor_mode_name,
        )
    _run_checkpoint_transfer_stage(config, split_manifest, dataset)


def main() -> None:
    run_configured_stage_cli(
        run_train_stage,
        description="Train MNIST through generated SPLL artifacts for Pipeline II.",
        config_help="Path to the Pipeline II YAML config.",
    )


if __name__ == "__main__":
    main()
