from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import queue
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import torch

from mnist_spll_common import TerminalProgressBar, load_config, save_config, set_seed, stage_message
from mnist_spll_pipeline_core import load_json, utc_now_iso, write_json
from spll_training_core import (
    DifferentiableReadMNIST,
    TensorLRUCache,
    assert_compiled_artifacts_exist,
    build_model_from_config,
    call_true_sum,
    cleanup_torch,
    compiled_program_path,
    generate_sum_case,
    get_experiments,
    get_inference_modes,
    get_seeds,
    grad_norm,
    initial_checkpoint_path,
    load_compiled_training_module,
    load_mnist_train_dataset,
    make_precomputed_read_mnist,
    recent_mean,
    run_dir,
    save_training_checkpoint,
    schedule_manifest_path,
    stable_int_seed,
    tensor_to_float,
    training_paths,
    validate_probability_and_loss,
    write_csv_header,
    write_failure_report,
    zero_anchor_from_model,
)
from mnist_spll_common import ensure_dir, resolve_device


def _load_initial_model(config: Dict[str, Any], checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = build_model_from_config(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model




def _cpu_clone(value: Any) -> Any:
    """Return a pickle-safe CPU snapshot of nested torch state."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


def _snapshot_model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}


def _snapshot_optimizer_state(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    return _cpu_clone(optimizer.state_dict())


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _checkpoint_transfer_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = config.get("checkpoint_transfer", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("checkpoint_transfer must be a mapping when provided.")
    enabled = bool(raw.get("enabled", True))
    anchor_mode_name = str(raw.get("anchor_mode_name", "exact")).strip() or "exact"
    configured_mode_names = raw.get("mode_names")
    if configured_mode_names is None:
        mode_names: Optional[List[str]] = None
    else:
        if not isinstance(configured_mode_names, list) or not all(str(v).strip() for v in configured_mode_names):
            raise ValueError("checkpoint_transfer.mode_names must be a list of non-empty mode names when provided.")
        mode_names = [str(v).strip() for v in configured_mode_names]
    include_final = bool(raw.get("include_final_checkpoint", True))
    return {
        "enabled": enabled,
        "anchor_mode_name": anchor_mode_name,
        "mode_names": mode_names,
        "include_final_checkpoint": include_final,
    }


def _checkpoint_transfer_run_dir(paths: Any, seed: int, n_terms: int, mode_name: str, anchor_mode_name: str) -> Path:
    return paths.runs_root / f"seed_{int(seed)}_terms_{int(n_terms):02d}_transfer_{mode_name}_from_{anchor_mode_name}"


def _checkpoint_transfer_checkpoint_path(run_dir_path: Path, segment_index: int) -> Path:
    return run_dir_path / "checkpoints" / f"segment_{int(segment_index):02d}.pt"


def _load_training_checkpoint(config: Dict[str, Any], checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = build_model_from_config(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    optimizer = _make_optimizer(config, model)
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device(optimizer, device)
    return model, optimizer, payload


def _threshold_key(value: float) -> str:
    return f"{float(value):.2f}"


def _threshold_label(value: float) -> str:
    return _threshold_key(value).replace(".", "p")


def _checkpointing_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = config.get("checkpointing", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("checkpointing must be a mapping when provided.")
    legacy_validation_cfg = config.get("validation", {}) or {}
    thresholds_raw = raw.get("posterior_thresholds", legacy_validation_cfg.get("milestones", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
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


def _aggregate_checkpoint_path(paths: Any, n_terms: int, anchor_mode_name: str) -> Path:
    return paths.root / "aggregate_checkpoints" / f"terms_{int(n_terms):02d}_{anchor_mode_name}_posterior_checkpoints.json"


def _mean_float(values: Sequence[float]) -> Optional[float]:
    cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not cleaned:
        return None
    return float(sum(cleaned) / len(cleaned))


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
                missing = [path for path in paths_by_seed.values() if not Path(path).exists()]
                if missing:
                    raise FileNotFoundError(
                        "Missing exact step checkpoint(s) needed for aggregate checkpoint transfer:\n"
                        + "\n".join(missing[:20])
                    )
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
    target_path = _aggregate_checkpoint_path(paths, n_terms, anchor_mode_name)
    ensure_dir(target_path.parent)
    write_json(target_path, payload)
    return payload


def _write_posterior_checkpoints_json(path: Path, checkpoints_state: Dict[str, Dict[str, Any]]) -> None:
    write_json(path, {"posterior_checkpoints": checkpoints_state})


def _checkpoint_milestone_label_candidates(milestone_value: float, raw_label: str) -> List[str]:
    candidates = [
        str(raw_label),
        str(float(milestone_value)),
        f"{float(milestone_value):.1f}",
        f"{float(milestone_value):.2f}",
        f"{float(milestone_value):g}",
    ]
    unique: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _resolve_milestone_checkpoint_path(exact_dir: Path, milestone_key: str) -> Path:
    """Resolve historical milestone checkpoint names robustly.

    ``milestones.json`` / ``run_summary.json`` stores milestone keys using
    ``f"{m:.2f}"`` (for example ``0.40``), while older checkpoint writers used
    ``str(m)`` from the float configured milestone (for example ``0.4``).  Both
    names refer to the same milestone.  This resolver accepts either spelling,
    plus any matching checkpoint file already present on disk.
    """

    checkpoint_dir = exact_dir / "checkpoints"
    milestone_value = float(milestone_key)
    for label in _checkpoint_milestone_label_candidates(milestone_value, str(milestone_key)):
        candidate = checkpoint_dir / f"milestone_{label.replace('.', 'p')}.pt"
        if candidate.exists():
            return candidate

    for candidate in checkpoint_dir.glob("milestone_*.pt"):
        raw_label = candidate.stem.removeprefix("milestone_").replace("p", ".")
        try:
            if math.isclose(float(raw_label), milestone_value, rel_tol=0.0, abs_tol=1.0e-12):
                return candidate
        except ValueError:
            continue

    # Return the canonical current-writer spelling so a later error message points
    # to the most likely expected file if the checkpoint is genuinely absent.
    canonical_label = f"{milestone_value:g}"
    return checkpoint_dir / f"milestone_{canonical_label.replace('.', 'p')}.pt"


def _checkpoint_transfer_anchor_sequence(
    config: Dict[str, Any],
    *,
    seed: int,
    n_terms: int,
    anchor_mode_name: str,
    include_final_checkpoint: bool,
) -> List[Dict[str, Any]]:
    paths = training_paths(config)
    aggregate_path = _aggregate_checkpoint_path(paths, n_terms, anchor_mode_name)
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
    target_dir = _checkpoint_transfer_run_dir(paths, seed, n_terms, mode_name, anchor_mode_name)
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
        "adaptive_top_k": bool(mode.get("adaptive_top_k", False)),
        "posterior_mass_target": mode.get("posterior_mass_target"),
    }

    try:
        segment_handle, segment_writer = write_csv_header(
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
        train_trace_handle, train_trace_writer = write_csv_header(
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
                "segment_local_cases",
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
                "grad_norm",
                "top_k_cutoff_runtime",
                "posterior_mass_target",
                "cutoff_search_mean_surviving_posterior_mass",
                "cutoff_search_abs_error",
            ],
        )

        checkpoint_cfg = _checkpointing_cfg(config)
        rolling_window = int(checkpoint_cfg["rolling_window_updates"])
        train_cfg = config.get("training", {})
        loss_epsilon = float(train_cfg.get("loss_epsilon", 1.0e-12))
        sum_batch_size = max(1, int(train_cfg.get("sum_batch_size", 1)))
        device = resolve_device(str(train_cfg.get("device", "auto")), bool(train_cfg.get("require_mps", False)))
        data_cfg = config.get("data", {})

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
            cache = TensorLRUCache(
                dataset,
                device=device,
                cache_device=str(data_cfg.get("image_cache_device", "device")),
                max_items=int(data_cfg.get("image_cache_max_items", 4096)),
                strategy=str(data_cfg.get("image_cache_strategy", "lru")),
            )
            read_mnist = DifferentiableReadMNIST(model, cache, device)
            setattr(module, "readMNist", read_mnist)
            adaptive_cutoff_state = _tune_adaptive_top_k_cutoff(
                config=config,
                mode=mode,
                module=module,
                model=model,
                cache=cache,
                device=device,
                split_manifest=split_manifest,
                seed=seed,
                n_terms=n_terms,
                step=start_step,
                reason="checkpoint_transfer_preflight",
            )
            setattr(module, "readMNist", read_mnist)
            _run_preflight(
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
            optimizer_update = 0
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
                batch_stats = _train_sum_batch(
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
                branch_count_mean = batch_stats["branch_count_mean"]
                branch_count_total = batch_stats["branch_count_total"]
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
                        "segment_local_cases": cases_seen - start_step,
                        "segment_elapsed_seconds": elapsed,
                        "loss": last_loss,
                        "true_mass": last_true_mass,
                        "zero_true_mass": last_zero_rate,
                        "loss_recent_mean": recent_mean(list(recent_losses)),
                        "true_mass_recent_mean": recent_mean(list(recent_masses)),
                        "zero_true_mass_recent_rate": recent_mean(list(recent_zeros)),
                        "branch_count": branch_count_mean,
                        "branch_count_mean": branch_count_mean,
                        "branch_count_total": branch_count_total,
                        "grad_norm": last_grad_norm,
                        "top_k_cutoff_runtime": _get_runtime_top_k_cutoff(module, mode),
                        "posterior_mass_target": mode.get("posterior_mass_target"),
                        "cutoff_search_mean_surviving_posterior_mass": (adaptive_cutoff_state or {}).get("mean_surviving_posterior_mass"),
                        "cutoff_search_abs_error": (adaptive_cutoff_state or {}).get("abs_error"),
                    }
                )
                if optimizer_update % 25 == 0 or cases_seen >= end_step:
                    train_trace_handle.flush()

            elapsed = time.perf_counter() - started_at
            rolling_loss = recent_mean(list(recent_losses))
            rolling_true_mass = recent_mean(list(recent_masses))
            checkpoint_path = _checkpoint_transfer_checkpoint_path(target_dir, segment_index)
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=end_step,
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
                "segment_cases": end_step - start_step,
                "segment_elapsed_seconds": elapsed,
                "end_loss_transfer": last_loss,
                "end_true_mass_transfer": last_true_mass,
                "end_loss_recent_mean_transfer": rolling_loss,
                "end_true_mass_recent_mean_transfer": rolling_true_mass,
                "end_zero_true_mass_recent_rate_transfer": recent_mean(list(recent_zeros)),
                "top_k_cutoff_runtime": _get_runtime_top_k_cutoff(module, mode),
                "posterior_mass_target": mode.get("posterior_mass_target"),
            }
            segment_rows.append(row)
            segment_writer.writerow(row)
            segment_handle.flush()
            cache.clear()
            cleanup_torch()
            progress.update(postfix=f"to step {end_step}, p={float(rolling_true_mass or 0.0):.3f}")

        progress.finish(postfix="done")
        segment_handle.close()
        train_trace_handle.close()
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
        cleanup_torch()
        raise

def _run_checkpoint_transfer_stage(config: Dict[str, Any], split_manifest: Dict[str, Any], dataset: Any) -> None:
    cfg = _checkpoint_transfer_cfg(config)
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


def _save_training_checkpoint_snapshot(
    path: Path,
    *,
    state_dict: Dict[str, torch.Tensor],
    optimizer_state_dict: Dict[str, Any],
    step: int,
    elapsed_seconds: float,
    digit_accuracy_value: float,
    config: Dict[str, Any],
    seed: int,
    n_terms: int,
    mode: Dict[str, Any],
    validation_metric_name: str = "sum_posterior_accuracy",
    sum_posterior_accuracy: Optional[float] = None,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "state_dict": state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "step": int(step),
            "elapsed_seconds": float(elapsed_seconds),
            "digit_accuracy": float(digit_accuracy_value),
            "validation_metric_name": str(validation_metric_name),
            "sum_posterior_accuracy": (float(sum_posterior_accuracy) if sum_posterior_accuracy is not None else float(digit_accuracy_value)),
            "model_config": dict(config.get("model", {})),
            "seed": int(seed),
            "n_terms": int(n_terms),
            "inference_mode": mode["name"],
            "artifact_name": mode.get("artifact_name", mode["name"]),
            "top_k_cutoff": mode.get("top_k_cutoff"),
            "adaptive_top_k": bool(mode.get("adaptive_top_k", False)),
            "posterior_mass_target": mode.get("posterior_mass_target"),
            "cutoff_search": mode.get("cutoff_search"),
            "config_path": str(config.get("_config_path", "")),
            "created_at_utc": utc_now_iso(),
        },
        path,
    )


def _run_validation_snapshot(
    *,
    config: Dict[str, Any],
    split_manifest: Dict[str, Any],
    seed: int,
    n_terms: int,
    mode: Dict[str, Any],
    model_state_dict: Dict[str, torch.Tensor],
    step: int,
    elapsed_seconds: float,
    posterior_cases: int,
    zero_total_mass_tolerance: float,
    device_name: str,
) -> Dict[str, Any]:
    device = resolve_device(str(device_name), False)
    model = build_model_from_config(config)
    model.load_state_dict(model_state_dict)
    model.to(device)
    dataset = load_mnist_train_dataset(config)

    paths = training_paths(config)
    module = load_compiled_training_module(compiled_program_path(paths, n_terms, mode), n_terms, str(mode["name"]))
    data_cfg = config.get("data", {})
    cache = TensorLRUCache(
        dataset,
        device=device,
        cache_device=str(data_cfg.get("image_cache_device", "device")),
        max_items=int(data_cfg.get("image_cache_max_items", 4096)),
        strategy=str(data_cfg.get("image_cache_strategy", "lru")),
    )
    cutoff_state = _tune_adaptive_top_k_cutoff(
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
        reason="async_validation",
    )
    metrics = _evaluate_sum_posterior_validation(
        module=module,
        model=model,
        cache=cache,
        device=device,
        split_manifest=split_manifest,
        seed=seed,
        n_terms=n_terms,
        num_cases=posterior_cases,
        zero_total_mass_tolerance=zero_total_mass_tolerance,
    )
    if cutoff_state is not None:
        metrics.update({f"cutoff_search_{key}": value for key, value in cutoff_state.items() if key != "evaluations"})
        metrics["top_k_cutoff_runtime"] = cutoff_state["runtime_top_k_cutoff"]
        metrics["posterior_mass_target"] = cutoff_state["posterior_mass_target"]
    else:
        metrics["top_k_cutoff_runtime"] = _get_runtime_top_k_cutoff(module, mode)
        metrics["posterior_mass_target"] = mode.get("posterior_mass_target")
    cache.clear()

    cleanup_torch()
    accuracy = float(metrics["sum_posterior_accuracy"])
    return {
        "step": int(step),
        "elapsed_seconds": float(elapsed_seconds),
        # Backward-compatible alias: existing milestone and visualization code reads
        # digit_accuracy, but the value is now task-level full-posterior sum accuracy.
        "digit_accuracy": accuracy,
        **metrics,
    }

def _validation_worker_main(
    request_queue: Any,
    result_queue: Any,
    *,
    config: Dict[str, Any],
    split_manifest: Dict[str, Any],
    seed: int,
    n_terms: int,
    mode: Dict[str, Any],
    posterior_cases: int,
    zero_total_mass_tolerance: float,
    device_name: str,
) -> None:
    try:
        torch.set_num_threads(max(1, int(config.get("validation", {}).get("async", {}).get("torch_num_threads", 1))))
    except Exception:
        pass
    while True:
        request = request_queue.get()
        if request is None or request.get("type") == "shutdown":
            return
        job_id = int(request["job_id"])
        try:
            result = _run_validation_snapshot(
                config=config,
                split_manifest=split_manifest,
                seed=seed,
                n_terms=n_terms,
                mode=mode,
                model_state_dict=request["model_state_dict"],
                step=int(request["step"]),
                elapsed_seconds=float(request["elapsed_seconds"]),
                posterior_cases=posterior_cases,
                zero_total_mass_tolerance=zero_total_mass_tolerance,
                device_name=device_name,
            )
            result.update(
                {
                    "type": "result",
                    "job_id": job_id,
                    "validation_snapshot_step": int(request.get("validation_snapshot_step", request.get("step", -1))),
                    "trainer_step_when_submitted": request.get("trainer_step_when_submitted"),
                    "loss_recent_mean": request.get("loss_recent_mean"),
                    "true_mass_recent_mean": request.get("true_mass_recent_mean"),
                    "zero_true_mass_recent_rate": request.get("zero_true_mass_recent_rate"),
                    "validation_source": "async_worker",
                }
            )
            result_queue.put(result)
        except BaseException as exc:
            result_queue.put(
                {
                    "type": "error",
                    "job_id": job_id,
                    "step": int(request.get("step", -1)),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

def _make_optimizer(config: Dict[str, Any], model: torch.nn.Module):
    opt_cfg = config.get("optimizer", {})
    name = str(opt_cfg.get("name", "adam")).lower()
    lr = float(opt_cfg.get("learning_rate", 0.001))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    if name != "adam":
        raise ValueError(f"Pipeline II currently supports optimizer.name=adam only, got {name!r}.")
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _run_preflight(
    *,
    module: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    split_manifest: Dict[str, Any],
    seed: int,
    n_terms: int,
    loss_epsilon: float,
) -> None:
    model.train()
    max_probe_steps = 50
    last_zero_mass = False
    for probe_step in range(1, max_probe_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        case = generate_sum_case(
            split_manifest,
            base_seed=seed,
            n_terms=n_terms,
            step=probe_step,
            split="train",
        )
        p_true, _branch_count = call_true_sum(
            module,
            int(case["true_sum"]),
            case["global_indices"],
            allow_pruned_zero=True,
            zero_anchor=zero_anchor_from_model(model),
        )
        if not isinstance(p_true, torch.Tensor) or not p_true.requires_grad:
            raise RuntimeError(
                "Generated SPLL artifact returned a probability that is not a differentiable torch.Tensor."
            )
        loss = -torch.log(p_true + loss_epsilon)
        validate_probability_and_loss(p_true, loss, step=0)
        loss.backward()
        norm = grad_norm(model)
        if norm > 0.0:
            optimizer.zero_grad(set_to_none=True)
            return
        last_zero_mass = tensor_to_float(p_true) <= 0.0

    reason = (
        "only fully pruned zero-mass cases were observed"
        if last_zero_mass
        else "no nonzero gradient was produced"
    )
    raise RuntimeError(
        f"Preflight differentiability check failed after {max_probe_steps} probe cases: {reason}."
    )


def _evaluate_sum_probe(
    *,
    module: Any,
    split_manifest: Dict[str, Any],
    seed: int,
    n_terms: int,
    num_cases: int,
) -> Optional[float]:
    if num_cases <= 0:
        return None
    masses: List[float] = []
    with torch.no_grad():
        for probe_step in range(1, num_cases + 1):
            case = generate_sum_case(
                split_manifest,
                base_seed=seed + 104729,
                n_terms=n_terms,
                step=probe_step,
                split="validation",
            )
            p_true, _branch_count = call_true_sum(
                module,
                int(case["true_sum"]),
                case["global_indices"],
                allow_pruned_zero=True,
            )
            masses.append(tensor_to_float(p_true))
    return float(sum(masses) / len(masses)) if masses else None


def _candidate_sums_for_terms(n_terms: int) -> List[int]:
    return list(range(0, 9 * int(n_terms) + 1))


def _validate_probability_value(value: float, *, candidate_sum: int, case_step: int) -> None:
    if not math.isfinite(value):
        raise FloatingPointError(
            f"Invalid posterior probability for candidate sum {candidate_sum} at validation case {case_step}: {value}"
        )
    if value < 0.0:
        raise FloatingPointError(
            f"Negative posterior probability for candidate sum {candidate_sum} at validation case {case_step}: {value}"
        )


def _evaluate_sum_posterior_validation(
    *,
    module: Any,
    model: torch.nn.Module,
    cache: TensorLRUCache,
    device: torch.device,
    split_manifest: Dict[str, Any],
    seed: int,
    n_terms: int,
    num_cases: int,
    zero_total_mass_tolerance: float,
) -> Dict[str, Any]:
    """Evaluate task accuracy from the generated SPLL full posterior over sums.

    For each held-out sum case this enumerates every valid candidate sum
    0..9*n_terms through the generated artifact, predicts argmax posterior mass,
    and compares that prediction to the true sum.  Approximate modes are evaluated
    under their own compiled pruning behavior; if pruning removes every candidate
    and total posterior mass is effectively zero, the case is counted as
    incorrect rather than benefiting from an arbitrary zero tie.
    """

    num_cases = int(num_cases)
    if num_cases <= 0:
        raise ValueError("validation.sum_posterior.num_cases must be positive when posterior validation is used.")

    candidate_sums = _candidate_sums_for_terms(n_terms)
    zero_total_mass_tolerance = max(0.0, float(zero_total_mass_tolerance))
    tie_tolerance = 1.0e-12

    was_training = model.training
    previous_read_mnist = getattr(module, "readMNist", None)
    model.eval()

    correct = 0
    total = 0
    zero_total = 0
    ties = 0
    true_masses: List[float] = []
    pred_masses: List[float] = []
    total_masses: List[float] = []
    branch_counts: List[int] = []

    try:
        with torch.no_grad():
            for probe_step in range(1, num_cases + 1):
                case = generate_sum_case(
                    split_manifest,
                    base_seed=seed + 104729,
                    n_terms=n_terms,
                    step=probe_step,
                    split="train",
                )
                setattr(
                    module,
                    "readMNist",
                    make_precomputed_read_mnist(
                        model=model,
                        tensor_cache=cache,
                        device=device,
                        global_indices=case["global_indices"],
                    ),
                )

                masses: List[float] = []
                for candidate_sum in candidate_sums:
                    probability, branch_count = call_true_sum(
                        module,
                        int(candidate_sum),
                        case["global_indices"],
                        allow_pruned_zero=True,
                    )
                    value = tensor_to_float(probability)
                    _validate_probability_value(value, candidate_sum=int(candidate_sum), case_step=int(case["step"]))
                    masses.append(value)
                    if branch_count is not None:
                        branch_counts.append(int(branch_count))

                true_sum = int(case["true_sum"])
                true_index = true_sum - candidate_sums[0]
                true_mass = masses[true_index] if 0 <= true_index < len(masses) else 0.0
                total_mass = float(sum(masses))
                true_masses.append(float(true_mass))
                total_masses.append(total_mass)

                if total_mass <= zero_total_mass_tolerance:
                    zero_total += 1
                    pred_masses.append(0.0)
                    total += 1
                    continue

                max_mass = max(masses)
                best_indices = [idx for idx, value in enumerate(masses) if abs(value - max_mass) <= tie_tolerance]
                if len(best_indices) > 1:
                    ties += 1
                pred_sum = candidate_sums[best_indices[0]]
                pred_masses.append(float(max_mass))
                correct += int(int(pred_sum) == true_sum)
                total += 1
    finally:
        if previous_read_mnist is not None:
            setattr(module, "readMNist", previous_read_mnist)
        if was_training:
            model.train()

    accuracy = float(correct / total) if total else 0.0
    return {
        "sum_posterior_accuracy": accuracy,
        "sum_posterior_cases": int(total),
        "sum_posterior_candidate_count": len(candidate_sums),
        "sum_posterior_mean_true_mass": float(sum(true_masses) / len(true_masses)) if true_masses else None,
        "sum_posterior_mean_pred_mass": float(sum(pred_masses) / len(pred_masses)) if pred_masses else None,
        "sum_posterior_mean_total_mass": float(sum(total_masses) / len(total_masses)) if total_masses else None,
        "sum_posterior_zero_total_rate": float(zero_total / total) if total else None,
        "sum_posterior_tie_rate": float(ties / total) if total else None,
        "sum_posterior_branch_count_mean": (float(sum(branch_counts) / len(branch_counts)) if branch_counts else None),
    }



def _is_adaptive_top_k_mode(mode: Dict[str, Any]) -> bool:
    return bool(mode.get("adaptive_top_k")) and mode.get("posterior_mass_target") is not None


def _require_mutable_top_k_cutoff(module: Any, *, mode_name: str) -> None:
    if not hasattr(module, "TOP_K_CUTOFF"):
        raise RuntimeError(
            f"Adaptive top-k mode {mode_name!r} requires a generated SPLL artifact compiled with -k. "
            "The exact artifact does not expose TOP_K_CUTOFF. Re-run the Pipeline II compile stage."
        )


def _set_runtime_top_k_cutoff(module: Any, cutoff: float) -> None:
    cutoff = max(0.0, min(1.0, float(cutoff)))
    setattr(module, "TOP_K_CUTOFF", cutoff)


def _get_runtime_top_k_cutoff(module: Any, mode: Dict[str, Any]) -> Optional[float]:
    if hasattr(module, "TOP_K_CUTOFF"):
        try:
            return float(getattr(module, "TOP_K_CUTOFF"))
        except (TypeError, ValueError):
            return None
    cutoff = mode.get("top_k_cutoff")
    return None if cutoff is None else float(cutoff)


def _mean_surviving_posterior_mass_for_cutoff(
    *,
    module: Any,
    model: torch.nn.Module,
    cache: TensorLRUCache,
    device: torch.device,
    split_manifest: Dict[str, Any],
    seed: int,
    n_terms: int,
    num_cases: int,
    cutoff: float,
) -> float:
    """Return mean unnormalised posterior mass surviving under one runtime cutoff.

    The generated approximate artifact returns raw, unnormalised mass.  Summing it
    over all candidate sums therefore estimates how much of the exact posterior
    survived pruning.  The redesigned Pipeline II benchmark does not use held-out
    validation during training, so adaptive probes use deterministic train-pool
    cases instead.
    """

    candidate_sums = _candidate_sums_for_terms(n_terms)
    previous_cutoff = getattr(module, "TOP_K_CUTOFF", None)
    previous_read_mnist = getattr(module, "readMNist", None)
    was_training = model.training
    _set_runtime_top_k_cutoff(module, cutoff)
    model.eval()
    total_masses: List[float] = []
    try:
        with torch.no_grad():
            for probe_step in range(1, int(num_cases) + 1):
                case = generate_sum_case(
                    split_manifest,
                    base_seed=seed + 104729,
                    n_terms=n_terms,
                    step=probe_step,
                    split="train",
                )
                setattr(
                    module,
                    "readMNist",
                    make_precomputed_read_mnist(
                        model=model,
                        tensor_cache=cache,
                        device=device,
                        global_indices=case["global_indices"],
                    ),
                )
                mass = 0.0
                for candidate_sum in candidate_sums:
                    probability, _branch_count = call_true_sum(
                        module,
                        int(candidate_sum),
                        case["global_indices"],
                        allow_pruned_zero=True,
                    )
                    value = tensor_to_float(probability)
                    _validate_probability_value(value, candidate_sum=int(candidate_sum), case_step=int(case["step"]))
                    mass += float(value)
                total_masses.append(float(mass))
    finally:
        if previous_cutoff is not None:
            setattr(module, "TOP_K_CUTOFF", previous_cutoff)
        if previous_read_mnist is not None:
            setattr(module, "readMNist", previous_read_mnist)
        if was_training:
            model.train()
    return float(sum(total_masses) / len(total_masses)) if total_masses else 0.0


def _tune_adaptive_top_k_cutoff(
    *,
    config: Dict[str, Any],
    mode: Dict[str, Any],
    module: Any,
    model: torch.nn.Module,
    cache: TensorLRUCache,
    device: torch.device,
    split_manifest: Dict[str, Any],
    seed: int,
    n_terms: int,
    step: int,
    reason: str,
) -> Optional[Dict[str, Any]]:
    """Tune TOP_K_CUTOFF to hit a target surviving posterior mass.

    The search is bounded and deterministic rather than simulated annealing:
    increasing TOP_K_CUTOFF should only prune more branches, so the objective is
    effectively one-dimensional and monotone.  This is easier to reproduce and
    easier to defend in the experimental methodology.
    """

    if not _is_adaptive_top_k_mode(mode):
        return None
    _require_mutable_top_k_cutoff(module, mode_name=str(mode["name"]))

    target = float(mode.get("posterior_mass_target", 0.8))
    search_cfg = dict(mode.get("cutoff_search") or {})
    probe_cases = max(1, int(search_cfg.get("probe_cases", 20)))
    max_iterations = max(1, int(search_cfg.get("max_iterations", 14)))
    tolerance = max(0.0, float(search_cfg.get("tolerance", 0.02)))
    low = max(0.0, min(1.0, float(search_cfg.get("min_cutoff", 0.0))))
    high = max(low, min(1.0, float(search_cfg.get("max_cutoff", 1.0))))

    def evaluate(cutoff: float) -> float:
        return _mean_surviving_posterior_mass_for_cutoff(
            module=module,
            model=model,
            cache=cache,
            device=device,
            split_manifest=split_manifest,
            seed=seed,
            n_terms=n_terms,
            num_cases=probe_cases,
            cutoff=float(cutoff),
        )

    evaluations: List[Dict[str, float]] = []
    low_mass = evaluate(low)
    evaluations.append({"cutoff": float(low), "mean_surviving_mass": float(low_mass)})
    best_cutoff = float(low)
    best_mass = float(low_mass)
    best_error = abs(best_mass - target)

    # If cutoff=0 already removes too much mass, no larger cutoff can repair it.
    if low_mass <= target or target >= 1.0:
        _set_runtime_top_k_cutoff(module, best_cutoff)
        return {
            "adaptive_top_k": True,
            "posterior_mass_target": float(target),
            "runtime_top_k_cutoff": float(best_cutoff),
            "mean_surviving_posterior_mass": float(best_mass),
            "abs_error": float(best_error),
            "probe_cases": int(probe_cases),
            "iterations": 0,
            "converged": bool(best_error <= tolerance),
            "reason": str(reason),
            "step": int(step),
            "search_method": "bounded_monotone_bisection",
            "evaluations": evaluations,
        }

    high_mass = evaluate(high)
    evaluations.append({"cutoff": float(high), "mean_surviving_mass": float(high_mass)})
    if abs(high_mass - target) < best_error:
        best_cutoff = float(high)
        best_mass = float(high_mass)
        best_error = abs(best_mass - target)

    # If even the maximum cutoff keeps too much mass, use the maximum tested cutoff.
    if high_mass >= target:
        _set_runtime_top_k_cutoff(module, best_cutoff)
        return {
            "adaptive_top_k": True,
            "posterior_mass_target": float(target),
            "runtime_top_k_cutoff": float(best_cutoff),
            "mean_surviving_posterior_mass": float(best_mass),
            "abs_error": float(best_error),
            "probe_cases": int(probe_cases),
            "iterations": 0,
            "converged": bool(best_error <= tolerance),
            "reason": str(reason),
            "step": int(step),
            "search_method": "bounded_monotone_bisection",
            "evaluations": evaluations,
        }

    iterations = 0
    left = float(low)
    right = float(high)
    for iterations in range(1, max_iterations + 1):
        mid = (left + right) / 2.0
        mid_mass = evaluate(mid)
        evaluations.append({"cutoff": float(mid), "mean_surviving_mass": float(mid_mass)})
        mid_error = abs(float(mid_mass) - target)
        if mid_error < best_error:
            best_cutoff = float(mid)
            best_mass = float(mid_mass)
            best_error = float(mid_error)
        if mid_error <= tolerance:
            break
        if mid_mass > target:
            left = mid
        else:
            right = mid

    _set_runtime_top_k_cutoff(module, best_cutoff)
    return {
        "adaptive_top_k": True,
        "posterior_mass_target": float(target),
        "runtime_top_k_cutoff": float(best_cutoff),
        "mean_surviving_posterior_mass": float(best_mass),
        "abs_error": float(best_error),
        "probe_cases": int(probe_cases),
        "iterations": int(iterations),
        "converged": bool(best_error <= tolerance),
        "reason": str(reason),
        "step": int(step),
        "search_method": "bounded_monotone_bisection",
        "evaluations": evaluations,
    }

def _train_sum_batch(
    *,
    module: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cache: TensorLRUCache,
    device: torch.device,
    cases: List[Dict[str, Any]],
    loss_epsilon: float,
) -> Dict[str, Any]:
    if not cases:
        raise ValueError("Cannot train on an empty SPLL sum batch.")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    all_indices: List[int] = []
    for case in cases:
        all_indices.extend(int(v) for v in case["global_indices"])
    setattr(
        module,
        "readMNist",
        make_precomputed_read_mnist(
            model=model,
            tensor_cache=cache,
            device=device,
            global_indices=all_indices,
        ),
    )

    zero_anchor = zero_anchor_from_model(model)
    p_trues: List[torch.Tensor] = []
    branch_counts: List[Optional[int]] = []
    for case in cases:
        p_true, branch_count = call_true_sum(
            module,
            int(case["true_sum"]),
            case["global_indices"],
            allow_pruned_zero=True,
            zero_anchor=zero_anchor,
        )
        if not isinstance(p_true, torch.Tensor) or not p_true.requires_grad:
            raise RuntimeError(f"Detached/non-tensor probability returned at training case {case['step']}.")
        p_trues.append(p_true)
        branch_counts.append(branch_count)

    p_true_batch = torch.stack(p_trues)
    per_case_losses = -torch.log(p_true_batch + loss_epsilon)
    loss = per_case_losses.mean()
    for case, p_true, case_loss in zip(cases, p_trues, per_case_losses):
        validate_probability_and_loss(p_true, case_loss, step=int(case["step"]))
    validate_probability_and_loss(p_true_batch.mean(), loss, step=int(cases[-1]["step"]))

    loss.backward()
    norm = grad_norm(model)
    optimizer.step()

    p_values = [tensor_to_float(value) for value in p_trues]
    loss_values = [tensor_to_float(value) for value in per_case_losses]
    zero_flags = [int(value <= 0.0) for value in p_values]
    numeric_branch_counts = [int(value) for value in branch_counts if value is not None]
    branch_count_mean = (
        float(sum(numeric_branch_counts) / len(numeric_branch_counts))
        if numeric_branch_counts
        else None
    )
    branch_count_total = int(sum(numeric_branch_counts)) if numeric_branch_counts else None
    return {
        "batch_size": len(cases),
        "loss": float(sum(loss_values) / len(loss_values)),
        "true_mass": float(sum(p_values) / len(p_values)),
        "zero_true_mass": float(sum(zero_flags) / len(zero_flags)),
        "branch_count_mean": branch_count_mean,
        "branch_count_total": branch_count_total,
        "grad_norm": norm,
        "case_loss_values": loss_values,
        "case_true_mass_values": p_values,
        "case_zero_flags": zero_flags,
    }


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
    validation_process = None
    validation_request_queue = None
    try:
        set_seed(seed)
        train_cfg = config.get("training", {})
        device = resolve_device(str(train_cfg.get("device", "auto")), bool(train_cfg.get("require_mps", False)))
        model = _load_initial_model(config, initial_checkpoint_path(paths, seed, n_terms), device)
        optimizer = _make_optimizer(config, model)

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
        adaptive_cutoff_state: Optional[Dict[str, Any]] = _tune_adaptive_top_k_cutoff(
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
        _run_preflight(
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
        transfer_cfg = _checkpoint_transfer_cfg(config)
        save_exact_step_checkpoints = (
            bool(transfer_cfg.get("enabled", True))
            and mode_name == str(transfer_cfg.get("anchor_mode_name", "exact"))
        )

        train_handle, train_writer = write_csv_header(
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
                "grad_norm",
                "top_k_cutoff_runtime",
                "posterior_mass_target",
                "cutoff_search_mean_surviving_posterior_mass",
                "cutoff_search_abs_error",
            ],
        )
        topk_event_handle, topk_event_writer = write_csv_header(
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
        )
        topk_search_handle, topk_search_writer = write_csv_header(
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
        )

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
            cutoff_state = _tune_adaptive_top_k_cutoff(
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
            batch_stats = _train_sum_batch(
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
                    "grad_norm": last_grad_norm,
                    "top_k_cutoff_runtime": _get_runtime_top_k_cutoff(module, mode),
                    "posterior_mass_target": mode.get("posterior_mass_target"),
                    "cutoff_search_mean_surviving_posterior_mass": (adaptive_cutoff_state or {}).get("mean_surviving_posterior_mass"),
                    "cutoff_search_abs_error": (adaptive_cutoff_state or {}).get("abs_error"),
                }
            )

            if save_exact_step_checkpoints:
                save_training_checkpoint(
                    _exact_step_checkpoint_path(this_run_dir, cases_seen),
                    model=model,
                    optimizer=optimizer,
                    step=cases_seen,
                    elapsed_seconds=elapsed,
                    digit_accuracy_value=float(rolling_true_mass if rolling_true_mass is not None else last_true_mass),
                    config=config,
                    seed=seed,
                    n_terms=n_terms,
                    mode=mode,
                    validation_metric_name="exact_step_training_checkpoint",
                    sum_posterior_accuracy=float(rolling_true_mass if rolling_true_mass is not None else last_true_mass),
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
                        }
                        milestones_state[key] = {
                            "reached": True,
                            "step": int(cases_seen),
                            "elapsed_seconds": float(elapsed),
                            "digit_accuracy": float(rolling_true_mass),
                            "sum_posterior_accuracy": float(rolling_true_mass),
                            "validation_metric": "true_sum_posterior_recent_mean",
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
                "runtime_top_k_cutoff": _get_runtime_top_k_cutoff(module, mode),
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
                "milestones": milestones_state,
            },
        )
        train_handle.close()
        topk_event_handle.close()
        topk_search_handle.close()
        cache.clear()
        cleanup_torch()
    except BaseException as exc:
        if validation_process is not None:
            try:
                if validation_request_queue is not None:
                    validation_request_queue.put_nowait({"type": "shutdown"})
            except Exception:
                pass
            try:
                validation_process.terminate()
                validation_process.join(timeout=2.0)
            except Exception:
                pass
        write_failure_report(this_run_dir / "failure_report.json", error=exc, context=context)
        cleanup_torch()
        raise

def run_train_stage(config: Dict[str, Any]) -> None:
    paths = training_paths(config)
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
    transfer_cfg = _checkpoint_transfer_cfg(config)
    anchor_mode_name = str(transfer_cfg.get("anchor_mode_name", "exact"))
    for experiment in experiments:
        _build_aggregate_exact_checkpoints(
            config,
            n_terms=int(experiment["n_terms"]),
            anchor_mode_name=anchor_mode_name,
            include_final_checkpoint=bool(transfer_cfg.get("include_final_checkpoint", True)),
        )
    _run_checkpoint_transfer_stage(config, split_manifest, dataset)


def run_checkpoint_transfer_only_stage(config: Dict[str, Any]) -> None:
    paths = training_paths(config)
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
    transfer_cfg = _checkpoint_transfer_cfg(config)
    anchor_mode_name = str(transfer_cfg.get("anchor_mode_name", "exact"))
    for experiment in get_experiments(config):
        _build_aggregate_exact_checkpoints(
            config,
            n_terms=int(experiment["n_terms"]),
            anchor_mode_name=anchor_mode_name,
            include_final_checkpoint=bool(transfer_cfg.get("include_final_checkpoint", True)),
        )
    _run_checkpoint_transfer_stage(config, split_manifest, dataset)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train MNIST through generated SPLL artifacts for Pipeline II.")
    parser.add_argument("--config", required=True, help="Path to the Pipeline II YAML config.")
    args = parser.parse_args()
    run_train_stage(load_config(args.config))


if __name__ == "__main__":
    main()
