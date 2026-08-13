from __future__ import annotations

import csv
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import torch

from mnist_model import resolve_device, set_seed
from pipeline2_artifacts import load_compiled_training_module
from pipeline2_config import (
    PIPELINE2_READ_MNIST_POLICY,
    aggregate_checkpoint_path,
    checkpoint_transfer_checkpoint_path,
    checkpoint_transfer_run_dir,
    get_checkpoint_transfer_config,
    compiled_program_path,
    get_experiments,
    get_inference_modes,
    get_seeds,
    initial_checkpoint_path,
    run_dir,
    stable_int_seed,
    training_paths,
)
from pipeline2_data import build_model_from_config, generate_sum_case
from pipeline2_runtime import (
    DirectReadMNIST,
    cleanup_torch,
    get_runtime_top_k_cutoff,
    load_initial_model,
    make_optimizer,
    move_optimizer_state_to_device,
    open_csv_trace,
    recent_mean,
    run_preflight,
    save_training_checkpoint,
    train_sum_batch,
    write_failure_report,
)
from pipeline_support import TerminalProgressBar, ensure_dir, load_json, utc_now_iso, write_json


def _load_training_checkpoint(config: Dict[str, Any], checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = build_model_from_config(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    optimizer = make_optimizer(config, model)
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        move_optimizer_state_to_device(optimizer, device)
    return model, optimizer, payload


def _threshold_key(value: float) -> str:
    return f"{float(value):.2f}"


def _threshold_label(value: float) -> str:
    return _threshold_key(value).replace(".", "p")


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _checkpointing_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = config.get("checkpointing", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("checkpointing must be a mapping when provided.")
    thresholds_raw = raw.get(
        "posterior_thresholds",
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    thresholds = sorted({float(v) for v in thresholds_raw})
    for value in thresholds:
        if not (0.0 < value <= 1.0):
            raise ValueError(f"checkpointing.posterior_thresholds values must be in (0, 1], got {value}")
    window = max(1, int(raw.get("rolling_window_updates", raw.get("rolling_window", 50))))
    policy = str(raw.get("rolling_window_policy", "full")).strip().lower()
    if policy != "full":
        raise ValueError("checkpointing.rolling_window_policy currently supports only 'full'.")
    return {
        "enabled": bool(raw.get("enabled", True)),
        "metric": str(raw.get("metric", "true_sum_posterior")),
        "posterior_thresholds": thresholds,
        "rolling_window_updates": window,
        "rolling_window_policy": policy,
    }


def _posterior_checkpoint_path(run_dir_path: Path, threshold: float) -> Path:
    return run_dir_path / "checkpoints" / f"posterior_{_threshold_label(threshold)}.pt"


def _exact_step_checkpoint_path(run_dir_path: Path, step: int) -> Path:
    return run_dir_path / "checkpoints" / "steps" / f"step_{int(step):08d}.pt"


def _full_window_rolling_mean_float(values: Sequence[float], window: int) -> List[Optional[float]]:
    """Trailing rolling mean with a strict full-window policy.

    The first ``window - 1`` positions return ``None`` rather than using a
    shorter prefix average.  This keeps checkpoint detection and plots honest:
    every rolling value is based on exactly ``window`` observed updates.
    """

    max_window = max(1, int(window))
    if max_window <= 1:
        return [float(value) for value in values]

    running: Deque[float] = deque()
    total = 0.0
    result: List[Optional[float]] = []
    for raw_value in values:
        value = float(raw_value)
        running.append(value)
        total += value
        while len(running) > max_window:
            total -= running.popleft()
        if len(running) < max_window:
            result.append(None)
        else:
            result.append(float(total / max_window))
    return result


def _read_training_trace_numeric(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                step = int(float(row.get("step", "")))
                true_mass = float(row.get("true_mass", ""))
                loss = float(row.get("loss", ""))
            except (TypeError, ValueError):
                continue
            item = {
                "step": float(step),
                "true_mass": true_mass,
                "loss": loss,
            }
            try:
                item["elapsed_seconds"] = float(row.get("elapsed_seconds", ""))
            except (TypeError, ValueError):
                pass
            rows.append(item)
    return rows


def _build_aggregate_exact_checkpoints(
    config: Dict[str, Any],
    *,
    n_terms: int,
    anchor_mode_name: str,
    include_final_checkpoint: bool,
) -> Dict[str, Any]:
    """Compute exact checkpoints from the aggregate exact trace across seeds.

    Checkpoint-transfer segments are intentionally defined only after all exact
    seeds are available: first average the exact true-sum posterior at each
    training step across seeds, then apply the configured rolling window, then
    find threshold crossings on that displayed aggregate curve.
    """

    paths = training_paths(config)
    checkpoint_cfg = _checkpointing_cfg(config)
    thresholds = list(checkpoint_cfg["posterior_thresholds"])
    window = int(checkpoint_cfg["rolling_window_updates"])
    seeds = [int(seed) for seed in get_seeds(config)]

    per_seed_rows: Dict[int, List[Dict[str, float]]] = {}
    for seed in seeds:
        trace_path = run_dir(paths, seed, n_terms, anchor_mode_name) / "train_trace.csv"
        rows = _read_training_trace_numeric(trace_path)
        if not rows:
            raise FileNotFoundError(
                f"Missing/empty exact training trace for aggregate checkpoints: {trace_path}. "
                f"Run pure {anchor_mode_name!r} training first."
            )
        per_seed_rows[int(seed)] = rows

    by_step: Dict[int, Dict[str, List[float]]] = {}
    for rows in per_seed_rows.values():
        for row in rows:
            step = int(row["step"])
            by_step.setdefault(step, {"true_mass": [], "loss": [], "elapsed_seconds": []})
            by_step[step]["true_mass"].append(float(row["true_mass"]))
            by_step[step]["loss"].append(float(row["loss"]))
            if "elapsed_seconds" in row:
                by_step[step]["elapsed_seconds"].append(float(row["elapsed_seconds"]))

    # Use only steps observed for all seeds so the aggregate checkpoint curve is
    # the same object that is plotted as the mean pure-exact trace.
    complete_steps = [
        step for step in sorted(by_step)
        if len(by_step[step]["true_mass"]) == len(seeds) and len(by_step[step]["loss"]) == len(seeds)
    ]
    if not complete_steps:
        raise RuntimeError(
            f"No complete across-seed exact steps found for n_terms={n_terms}, mode={anchor_mode_name!r}."
        )

    mean_true_mass = [float(sum(by_step[step]["true_mass"]) / len(seeds)) for step in complete_steps]
    mean_loss = [float(sum(by_step[step]["loss"]) / len(seeds)) for step in complete_steps]
    rolling_true_mass = _full_window_rolling_mean_float(mean_true_mass, window)
    rolling_loss = _full_window_rolling_mean_float(mean_loss, window)

    trace_rows: List[Dict[str, Any]] = []
    for step, true_value, loss_value, rolling_true, rolling_loss_value in zip(
        complete_steps, mean_true_mass, mean_loss, rolling_true_mass, rolling_loss
    ):
        if rolling_true is None or rolling_loss_value is None:
            continue
        trace_rows.append(
            {
                "step": int(step),
                "mean_true_mass": float(true_value),
                "mean_loss": float(loss_value),
                "rolling_true_mass": float(rolling_true),
                "rolling_loss": float(rolling_loss_value),
                "seed_count": len(seeds),
                "rolling_window_observations_per_seed": int(window),
            }
        )
    if not trace_rows:
        raise RuntimeError(
            f"Not enough complete exact trace points for a full-window aggregate checkpoint curve: "
            f"n_terms={n_terms}, mode={anchor_mode_name!r}, window={window}, complete_steps={len(complete_steps)}."
        )

    checkpoints: Dict[str, Dict[str, Any]] = {
        _threshold_key(threshold): {
            "reached": False,
            "step": None,
            "rolling_true_mass": None,
            "rolling_loss": None,
            "threshold": float(threshold),
            "metric": "aggregate_exact_true_sum_posterior_full_window_rolling_mean",
            "checkpoint_paths_by_seed": {},
        }
        for threshold in thresholds
    }
    for threshold in thresholds:
        key = _threshold_key(threshold)
        for row in trace_rows:
            if float(row["rolling_true_mass"]) >= float(threshold):
                step = int(row["step"])
                paths_by_seed = {
                    str(seed): str(_exact_step_checkpoint_path(run_dir(paths, seed, n_terms, anchor_mode_name), step))
                    for seed in seeds
                }
                checkpoints[key] = {
                    "reached": True,
                    "step": step,
                    "rolling_true_mass": float(row["rolling_true_mass"]),
                    "rolling_loss": float(row["rolling_loss"]),
                    "threshold": float(threshold),
                    "metric": "aggregate_exact_true_sum_posterior_full_window_rolling_mean",
                    "checkpoint_paths_by_seed": paths_by_seed,
                }
                break

    anchors: List[Dict[str, Any]] = [
        {
            "label": "initial",
            "step": 0,
            "rolling_true_mass": None,
            "rolling_loss": None,
            "checkpoint_paths_by_seed": {
                str(seed): str(initial_checkpoint_path(paths, seed, n_terms))
                for seed in seeds
            },
        }
    ]
    for key in sorted(checkpoints, key=lambda value: float(value)):
        info = checkpoints[key]
        if not bool(info.get("reached", False)):
            continue
        anchors.append(
            {
                "label": f"posterior_{key}",
                "step": int(info["step"]),
                "rolling_true_mass": info.get("rolling_true_mass"),
                "rolling_loss": info.get("rolling_loss"),
                "threshold": float(key),
                "checkpoint_paths_by_seed": dict(info.get("checkpoint_paths_by_seed") or {}),
            }
        )
    if include_final_checkpoint:
        final_step = int(complete_steps[-1])
        final_row = trace_rows[-1]
        anchors.append(
            {
                "label": "final",
                "step": final_step,
                "rolling_true_mass": float(final_row["rolling_true_mass"]),
                "rolling_loss": float(final_row["rolling_loss"]),
                "checkpoint_paths_by_seed": {
                    str(seed): str(run_dir(paths, seed, n_terms, anchor_mode_name) / "checkpoints" / "final.pt")
                    for seed in seeds
                },
            }
        )

    deduped_anchors: List[Dict[str, Any]] = []
    for anchor in sorted(anchors, key=lambda item: (int(item["step"]), str(item["label"]))):
        if deduped_anchors and int(anchor["step"]) <= int(deduped_anchors[-1]["step"]):
            if int(anchor["step"]) == int(deduped_anchors[-1]["step"]):
                deduped_anchors[-1] = anchor
            continue
        deduped_anchors.append(anchor)

    payload = {
        "created_at_utc": utc_now_iso(),
        "n_terms": int(n_terms),
        "anchor_mode_name": str(anchor_mode_name),
        "seed_count": len(seeds),
        "seeds": seeds,
        "rolling_window_updates": int(window),
        "rolling_window_policy": "full",
        "thresholds": [float(value) for value in thresholds],
        "source": "aggregate_exact_trace_after_all_seeds_full_window",
        "trace": trace_rows,
        "posterior_checkpoints": checkpoints,
        "anchors": deduped_anchors,
    }
    target_path = aggregate_checkpoint_path(paths, n_terms, anchor_mode_name)
    ensure_dir(target_path.parent)
    write_json(target_path, payload)
    return payload


def _materialize_aggregate_exact_anchor_checkpoints(
    config: Dict[str, Any],
    split_manifest: Dict[str, Any],
    dataset: Any,
    *,
    n_terms: int,
    anchor_mode_name: str,
) -> None:
    """Create checkpoint-transfer anchor snapshots outside timed training.

    The timed exact run should measure generated-SPLL training, not per-step disk
    checkpoint I/O. Aggregate checkpoints are therefore first selected from the
    exact trace, then the exact run is replayed deterministically and only the
    selected anchor steps are persisted. This keeps checkpoint-transfer support
    without charging exact mode for writes that approximate modes do not need.
    """

    paths = training_paths(config)
    aggregate_path = aggregate_checkpoint_path(paths, n_terms, anchor_mode_name)
    if not aggregate_path.exists():
        return
    aggregate = load_json(aggregate_path)
    checkpoints = dict(aggregate.get("posterior_checkpoints") or {})
    requested_by_seed: Dict[int, Dict[int, Path]] = {}

    for checkpoint in checkpoints.values():
        if not isinstance(checkpoint, dict) or not checkpoint.get("reached"):
            continue
        raw_step = checkpoint.get("step")
        if raw_step is None:
            continue
        step = int(raw_step)
        if step <= 0:
            continue
        for raw_seed, raw_path in dict(checkpoint.get("checkpoint_paths_by_seed") or {}).items():
            path = Path(str(raw_path))
            if path.exists():
                continue
            requested_by_seed.setdefault(int(raw_seed), {})[step] = path

    if not requested_by_seed:
        return

    modes_by_name = {str(mode["name"]): mode for mode in get_inference_modes(config)}
    if anchor_mode_name not in modes_by_name:
        raise ValueError(f"Unknown checkpoint-transfer anchor mode: {anchor_mode_name!r}")
    mode = modes_by_name[anchor_mode_name]

    train_cfg = config.get("training", {})
    device = resolve_device(str(train_cfg.get("device", "auto")), bool(train_cfg.get("require_mps", False)))
    loss_epsilon = float(train_cfg.get("loss_epsilon", 1.0e-12))
    sum_batch_size = max(1, int(train_cfg.get("sum_batch_size", 1)))
    total_steps = sum(len(steps) for steps in requested_by_seed.values())
    print(
        f"Materializing {total_steps} exact anchor checkpoint(s) for "
        f"terms={n_terms}, mode={anchor_mode_name} outside timed training."
    )

    for seed, step_paths in sorted(requested_by_seed.items()):
        max_step = max(step_paths)
        set_seed(seed)
        model = load_initial_model(config, initial_checkpoint_path(paths, seed, n_terms), device)
        optimizer = make_optimizer(config, model)
        module = load_compiled_training_module(compiled_program_path(paths, n_terms, mode), n_terms, anchor_mode_name)
        read_mnist = DirectReadMNIST(model, dataset, device)
        setattr(module, "readMNist", read_mnist)
        run_preflight(
            module=module,
            model=model,
            optimizer=optimizer,
            split_manifest=split_manifest,
            seed=seed,
            n_terms=n_terms,
            loss_epsilon=loss_epsilon,
        )

        cases_seen = 0
        started_at = time.perf_counter()
        progress = TerminalProgressBar(
            max_step,
            desc=f"Replay exact anchors terms={n_terms} seed={seed}",
            unit="iterations",
            enabled=bool(config.get("show_progress", True)),
        )
        try:
            while cases_seen < max_step:
                case_start_step = cases_seen + 1
                case_end_step = min(max_step, cases_seen + sum_batch_size)
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
                batch_stats = train_sum_batch(
                    module=module,
                    model=model,
                    optimizer=optimizer,
                    read_mnist=read_mnist,
                    cases=batch_cases,
                    loss_epsilon=loss_epsilon,
                )
                cases_seen = case_end_step
                if cases_seen in step_paths:
                    save_training_checkpoint(
                        step_paths[cases_seen],
                        model=model,
                        optimizer=optimizer,
                        step=cases_seen,
                        elapsed_seconds=time.perf_counter() - started_at,
                        digit_accuracy_value=float(batch_stats["true_mass"]),
                        config=config,
                        seed=seed,
                        n_terms=n_terms,
                        mode=mode,
                        validation_metric_name="aggregate_anchor_replay",
                        sum_posterior_accuracy=float(batch_stats["true_mass"]),
                    )
                progress.update(batch_stats["batch_size"], postfix=f"step={cases_seen}")
            progress.finish(postfix="anchors materialized")
        finally:
            cleanup_torch()


def _write_posterior_checkpoints_json(path: Path, checkpoints_state: Dict[str, Dict[str, Any]]) -> None:
    write_json(path, {"posterior_checkpoints": checkpoints_state})


def _checkpoint_transfer_anchor_sequence(
    config: Dict[str, Any],
    *,
    seed: int,
    n_terms: int,
    anchor_mode_name: str,
    include_final_checkpoint: bool,
) -> List[Dict[str, Any]]:
    paths = training_paths(config)
    aggregate_path = aggregate_checkpoint_path(paths, n_terms, anchor_mode_name)
    if not aggregate_path.exists():
        _build_aggregate_exact_checkpoints(
            config,
            n_terms=n_terms,
            anchor_mode_name=anchor_mode_name,
            include_final_checkpoint=include_final_checkpoint,
        )
    payload = load_json(aggregate_path)
    anchors: List[Dict[str, Any]] = []
    for anchor in payload.get("anchors", []):
        paths_by_seed = dict(anchor.get("checkpoint_paths_by_seed") or {})
        checkpoint_path = paths_by_seed.get(str(int(seed)))
        if not checkpoint_path:
            raise FileNotFoundError(
                f"Aggregate checkpoint {aggregate_path} has no checkpoint path for seed={seed}."
            )
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Missing exact checkpoint-transfer anchor checkpoint for seed={seed}: {checkpoint_path}"
            )
        anchors.append(
            {
                "label": str(anchor.get("label", "anchor")),
                "step": int(anchor["step"]),
                "rolling_true_mass": anchor.get("rolling_true_mass"),
                "rolling_loss": anchor.get("rolling_loss"),
                "checkpoint_path": Path(checkpoint_path),
                "threshold": anchor.get("threshold"),
                "aggregate_checkpoint_source": str(aggregate_path),
            }
        )
    anchors.sort(key=lambda item: (int(item["step"]), str(item["label"])))
    if len(anchors) < 2:
        raise RuntimeError(
            f"Need at least two aggregate exact posterior anchors to run checkpoint-transfer trajectories for seed={seed}, n_terms={n_terms}."
        )
    return anchors


def _run_checkpoint_transfer_for_mode(
    *,
    config: Dict[str, Any],
    split_manifest: Dict[str, Any],
    dataset: Any,
    seed: int,
    n_terms: int,
    mode: Dict[str, Any],
    anchor_mode_name: str,
    include_final_checkpoint: bool,
) -> None:
    paths = training_paths(config)
    mode_name = str(mode["name"])
    target_dir = checkpoint_transfer_run_dir(paths, seed, n_terms, mode_name, anchor_mode_name)
    ensure_dir(target_dir)
    ensure_dir(target_dir / "checkpoints")
    stale_failure_report = target_dir / "failure_report.json"
    if stale_failure_report.exists():
        stale_failure_report.unlink()

    anchors = _checkpoint_transfer_anchor_sequence(
        config,
        seed=seed,
        n_terms=n_terms,
        anchor_mode_name=anchor_mode_name,
        include_final_checkpoint=include_final_checkpoint,
    )

    context = {
        "seed": int(seed),
        "n_terms": int(n_terms),
        "mode": mode_name,
        "anchor_mode_name": anchor_mode_name,
        "segments": len(anchors) - 1,
        "top_k_cutoff": mode.get("top_k_cutoff"),
        "read_mnist_policy": PIPELINE2_READ_MNIST_POLICY,
    }

    csv_stack = ExitStack()
    try:
        segment_handle, segment_writer = csv_stack.enter_context(open_csv_trace(
            target_dir / "checkpoint_transfer_trace.csv",
            [
                "segment_index",
                "seed",
                "n_terms",
                "mode_name",
                "anchor_mode_name",
                "anchor_label",
                "anchor_step",
                "anchor_rolling_true_mass_exact",
                "anchor_rolling_loss_exact",
                "target_label",
                "target_step",
                "target_rolling_true_mass_exact",
                "target_rolling_loss_exact",
                "target_posterior_stop_value",
                "max_segment_cases_exact",
                "actual_end_step",
                "segment_cases",
                "reached_target_checkpoint",
                "segment_elapsed_seconds",
                "end_loss_transfer",
                "end_true_mass_transfer",
                "end_loss_recent_mean_transfer",
                "end_true_mass_recent_mean_transfer",
                "end_zero_true_mass_recent_rate_transfer",
                "top_k_cutoff_runtime",
                "read_mnist_model_evaluations",
            ],
        ))
        train_trace_handle, train_trace_writer = csv_stack.enter_context(open_csv_trace(
            target_dir / "checkpoint_transfer_train_trace.csv",
            [
                "segment_index",
                "seed",
                "n_terms",
                "mode_name",
                "anchor_mode_name",
                "step",
                "optimizer_update",
                "batch_size",
                "case_start_step",
                "case_end_step",
                "segment_start_step",
                "segment_target_step",
                "segment_budget_cases",
                "segment_local_cases",
                "target_posterior_stop_value",
                "reached_target_checkpoint",
                "segment_elapsed_seconds",
                "loss",
                "true_mass",
                "zero_true_mass",
                "loss_recent_mean",
                "true_mass_recent_mean",
                "zero_true_mass_recent_rate",
                "branch_count",
                "branch_count_mean",
                "branch_count_total",
                "read_mnist_calls_total",
                "read_mnist_model_evaluations_total",
                "read_mnist_model_evaluations_cumulative",
                "grad_norm",
                "top_k_cutoff_runtime",
            ],
        ))

        checkpoint_cfg = _checkpointing_cfg(config)
        rolling_window = int(checkpoint_cfg["rolling_window_updates"])
        train_cfg = config.get("training", {})
        loss_epsilon = float(train_cfg.get("loss_epsilon", 1.0e-12))
        sum_batch_size = max(1, int(train_cfg.get("sum_batch_size", 1)))
        device = resolve_device(str(train_cfg.get("device", "auto")), bool(train_cfg.get("require_mps", False)))
        progress = TerminalProgressBar(
            len(anchors) - 1,
            desc=f"Transfer {anchor_mode_name}->{mode_name} seed={seed} terms={n_terms}",
            unit="segments",
            enabled=bool(config.get("show_progress", True)),
        )

        segment_rows: List[Dict[str, Any]] = []
        for segment_index, (start_anchor, end_anchor) in enumerate(zip(anchors[:-1], anchors[1:]), start=1):
            start_step = int(start_anchor["step"])
            end_step = int(end_anchor["step"])
            if end_step <= start_step:
                continue

            set_seed(stable_int_seed("checkpoint_transfer", seed, n_terms, mode_name, start_step, end_step))
            model, optimizer, _payload = _load_training_checkpoint(config, Path(start_anchor["checkpoint_path"]), device)
            module = load_compiled_training_module(compiled_program_path(paths, n_terms, mode), n_terms, f"transfer_{mode_name}_{segment_index}")
            read_mnist = DirectReadMNIST(model, dataset, device)
            setattr(module, "readMNist", read_mnist)
            run_preflight(
                module=module,
                model=model,
                optimizer=optimizer,
                split_manifest=split_manifest,
                seed=seed,
                n_terms=n_terms,
                loss_epsilon=loss_epsilon,
            )

            recent_losses: Deque[float] = deque(maxlen=max(1, rolling_window))
            recent_masses: Deque[float] = deque(maxlen=max(1, rolling_window))
            recent_zeros: Deque[float] = deque(maxlen=max(1, rolling_window))
            started_at = time.perf_counter()
            cases_seen = start_step
            max_segment_cases_exact = end_step - start_step
            # Every transfer segment must consume the same deterministic case
            # interval as its exact counterpart, all the way through end_step.
            # The next exact checkpoint's posterior value remains useful as a
            # diagnostic crossing target, but it must not terminate the segment:
            # a faster-rising continuation would otherwise end left of the next
            # checkpoint in the trajectory plot and would no longer be a
            # fixed-budget checkpoint-to-checkpoint comparison.
            target_posterior_stop_value: Optional[float] = None
            if end_anchor.get("threshold") is not None:
                # Keep the historical CSV field name for artifact compatibility.
                # Semantically this is now a reference/crossing value only.
                target_posterior_stop_value = _optional_float(end_anchor.get("rolling_true_mass"))
                if target_posterior_stop_value is None:
                    target_posterior_stop_value = _optional_float(end_anchor.get("threshold"))
            reached_target_checkpoint = False
            optimizer_update = 0
            cumulative_read_mnist_model_evaluations = 0
            last_loss: Optional[float] = None
            last_true_mass: Optional[float] = None
            last_zero_rate: Optional[float] = None
            last_grad_norm: Optional[float] = None
            while cases_seen < end_step:
                case_start_step = cases_seen + 1
                case_end_step = min(end_step, cases_seen + sum_batch_size)
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
                    read_mnist=read_mnist,
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
                reached_target_this_update = False
                if (
                    not reached_target_checkpoint
                    and target_posterior_stop_value is not None
                    and len(recent_masses) >= max(1, rolling_window)
                    and rolling_true_mass is not None
                    and float(rolling_true_mass) >= float(target_posterior_stop_value)
                ):
                    reached_target_checkpoint = True
                    reached_target_this_update = True
                branch_count_mean = batch_stats["branch_count_mean"]
                branch_count_total = batch_stats["branch_count_total"]
                read_mnist_calls_total = int(batch_stats["read_mnist_calls_total"] or 0)
                read_mnist_model_evaluations_total = int(
                    batch_stats["read_mnist_model_evaluations_total"] or 0
                )
                cumulative_read_mnist_model_evaluations += read_mnist_model_evaluations_total
                train_trace_writer.writerow(
                    {
                        "segment_index": segment_index,
                        "seed": int(seed),
                        "n_terms": int(n_terms),
                        "mode_name": mode_name,
                        "anchor_mode_name": anchor_mode_name,
                        "step": cases_seen,
                        "optimizer_update": optimizer_update,
                        "batch_size": batch_stats["batch_size"],
                        "case_start_step": case_start_step,
                        "case_end_step": case_end_step,
                        "segment_start_step": start_step,
                        "segment_target_step": end_step,
                        "segment_budget_cases": max_segment_cases_exact,
                        "segment_local_cases": cases_seen - start_step,
                        "target_posterior_stop_value": target_posterior_stop_value,
                        "reached_target_checkpoint": reached_target_checkpoint,
                        "segment_elapsed_seconds": elapsed,
                        "loss": last_loss,
                        "true_mass": last_true_mass,
                        "zero_true_mass": last_zero_rate,
                        "loss_recent_mean": rolling_loss,
                        "true_mass_recent_mean": rolling_true_mass,
                        "zero_true_mass_recent_rate": rolling_zero_rate,
                        "branch_count": branch_count_mean,
                        "branch_count_mean": branch_count_mean,
                        "branch_count_total": branch_count_total,
                        "read_mnist_calls_total": read_mnist_calls_total,
                        "read_mnist_model_evaluations_total": read_mnist_model_evaluations_total,
                        "read_mnist_model_evaluations_cumulative": cumulative_read_mnist_model_evaluations,
                        "grad_norm": last_grad_norm,
                        "top_k_cutoff_runtime": get_runtime_top_k_cutoff(module, mode),
                    }
                )
                if optimizer_update % 25 == 0 or cases_seen >= end_step or reached_target_this_update:
                    train_trace_handle.flush()

            elapsed = time.perf_counter() - started_at
            actual_end_step = cases_seen
            rolling_loss = recent_mean(list(recent_losses))
            rolling_true_mass = recent_mean(list(recent_masses))
            checkpoint_path = checkpoint_transfer_checkpoint_path(target_dir, segment_index)
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=actual_end_step,
                elapsed_seconds=elapsed,
                digit_accuracy_value=float(rolling_true_mass if rolling_true_mass is not None else (last_true_mass or 0.0)),
                config=config,
                seed=seed,
                n_terms=n_terms,
                mode=mode,
                validation_metric_name="true_sum_posterior_rolling_mean",
                sum_posterior_accuracy=float(rolling_true_mass if rolling_true_mass is not None else (last_true_mass or 0.0)),
            )
            row = {
                "segment_index": segment_index,
                "seed": int(seed),
                "n_terms": int(n_terms),
                "mode_name": mode_name,
                "anchor_mode_name": anchor_mode_name,
                "anchor_label": start_anchor["label"],
                "anchor_step": start_step,
                "anchor_rolling_true_mass_exact": start_anchor.get("rolling_true_mass"),
                "anchor_rolling_loss_exact": start_anchor.get("rolling_loss"),
                "target_label": end_anchor["label"],
                "target_step": end_step,
                "target_rolling_true_mass_exact": end_anchor.get("rolling_true_mass"),
                "target_rolling_loss_exact": end_anchor.get("rolling_loss"),
                "target_posterior_stop_value": target_posterior_stop_value,
                "max_segment_cases_exact": max_segment_cases_exact,
                "actual_end_step": actual_end_step,
                "segment_cases": actual_end_step - start_step,
                "reached_target_checkpoint": reached_target_checkpoint,
                "segment_elapsed_seconds": elapsed,
                "end_loss_transfer": last_loss,
                "end_true_mass_transfer": last_true_mass,
                "end_loss_recent_mean_transfer": rolling_loss,
                "end_true_mass_recent_mean_transfer": rolling_true_mass,
                "end_zero_true_mass_recent_rate_transfer": recent_mean(list(recent_zeros)),
                "top_k_cutoff_runtime": get_runtime_top_k_cutoff(module, mode),
                "read_mnist_model_evaluations": int(cumulative_read_mnist_model_evaluations),
            }
            segment_rows.append(row)
            segment_writer.writerow(row)
            segment_handle.flush()
            cleanup_torch()
            status_label = "reached" if reached_target_checkpoint else "budget"
            progress.update(postfix=f"{status_label} step {actual_end_step}/{end_step}, p={float(rolling_true_mass or 0.0):.3f}")

        progress.finish(postfix="done")
        write_json(
            target_dir / "run_summary.json",
            {
                **context,
                "segment_count": len(segment_rows),
                "segments": segment_rows,
            },
        )
    except BaseException as exc:
        write_failure_report(target_dir / "failure_report.json", error=exc, context=context)
        raise
    finally:
        csv_stack.close()
        cleanup_torch()


def _run_checkpoint_transfer_stage(config: Dict[str, Any], split_manifest: Dict[str, Any], dataset: Any) -> None:
    cfg = get_checkpoint_transfer_config(config)
    if not cfg.get("enabled", True):
        return
    anchor_mode_name = str(cfg["anchor_mode_name"])
    all_modes = get_inference_modes(config)
    mode_lookup = {str(mode["name"]): mode for mode in all_modes}
    requested_mode_names = cfg.get("mode_names")
    if requested_mode_names is None:
        target_modes = [mode for mode in all_modes if str(mode["name"]) != anchor_mode_name]
    else:
        missing = [name for name in requested_mode_names if name not in mode_lookup]
        if missing:
            raise ValueError(f"Unknown checkpoint_transfer mode name(s): {missing}")
        target_modes = [mode_lookup[name] for name in requested_mode_names if name != anchor_mode_name]
    if not target_modes:
        return

    total = len(get_seeds(config)) * len(get_experiments(config)) * len(target_modes)
    outer = TerminalProgressBar(total, desc="Checkpoint-transfer runs", unit="runs", enabled=bool(config.get("show_progress", True)))
    for seed in get_seeds(config):
        for experiment in get_experiments(config):
            n_terms = int(experiment["n_terms"])
            for mode in target_modes:
                _run_checkpoint_transfer_for_mode(
                    config=config,
                    split_manifest=split_manifest,
                    dataset=dataset,
                    seed=seed,
                    n_terms=n_terms,
                    mode=mode,
                    anchor_mode_name=anchor_mode_name,
                    include_final_checkpoint=bool(cfg.get("include_final_checkpoint", True)),
                )
                outer.update(postfix=f"seed={seed}, terms={n_terms}, mode={mode['name']}")
    outer.finish(postfix="done")
