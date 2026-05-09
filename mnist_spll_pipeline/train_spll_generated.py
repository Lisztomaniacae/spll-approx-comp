from __future__ import annotations

import json
import math
import multiprocessing as mp
import queue
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

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
            "top_k_cutoff": mode.get("top_k_cutoff"),
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
                    split="validation",
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
        "top_k_cutoff": mode.get("top_k_cutoff"),
        "mode_order_position": int(mode_order_position),
        "mode_order_offset": int(mode_order_offset),
        "run_order_index": None if run_order_index is None else int(run_order_index),
    }
    validation_process = None
    validation_request_queue = None
    try:
        set_seed(seed)
        device_cfg = config.get("training", {})
        device = resolve_device(str(device_cfg.get("device", "auto")), bool(device_cfg.get("require_mps", False)))
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

        train_cfg = config.get("training", {})
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

        validation_cfg = config.get("validation", {})
        validation_interval = max(1, int(validation_cfg.get("interval_steps", 100)))
        milestones = [float(v) for v in validation_cfg.get("milestones", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])]
        highest_milestone = max(milestones)
        stop_at_highest = bool(validation_cfg.get("stop_at_highest_milestone", True))
        posterior_cfg = validation_cfg.get("sum_posterior", {})
        legacy_sum_probe_cfg = validation_cfg.get("sum_probe", {})
        posterior_cases = int(posterior_cfg.get("num_cases", legacy_sum_probe_cfg.get("num_cases", 100)))
        zero_total_mass_tolerance = float(posterior_cfg.get("zero_total_mass_tolerance", 1.0e-12))
        async_cfg = validation_cfg.get("async", {})
        async_enabled = bool(async_cfg.get("enabled", False))
        async_device_name = str(async_cfg.get("device", "cpu"))
        async_max_pending = max(1, int(async_cfg.get("max_pending_jobs", 1)))
        async_validate_step_zero = bool(async_cfg.get("validate_step_zero_synchronously", True))
        async_join_timeout_sec = float(async_cfg.get("join_timeout_sec", 5.0))

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
                "branch_count",
                "branch_count_mean",
                "branch_count_total",
                "grad_norm",
            ],
        )
        val_handle, val_writer = write_csv_header(
            this_run_dir / "validation_trace.csv",
            [
                "step",
                "elapsed_seconds",
                "digit_accuracy",
                "sum_posterior_accuracy",
                "sum_posterior_cases",
                "sum_posterior_candidate_count",
                "sum_posterior_mean_true_mass",
                "sum_posterior_mean_pred_mass",
                "sum_posterior_mean_total_mass",
                "sum_posterior_zero_total_rate",
                "sum_posterior_tie_rate",
                "sum_posterior_branch_count_mean",
                "validation_metric",
                "loss_recent_mean",
                "true_mass_recent_mean",
                "zero_true_mass_recent_rate",
                "validation_source",
                "trainer_step_when_recorded",
                "validation_snapshot_step",
                "trainer_step_when_submitted",
                "validation_lag_steps",
            ],
        )

        milestones_state: Dict[str, Dict[str, Any]] = {
            f"{m:.2f}": {"reached": False, "step": None, "elapsed_seconds": None, "digit_accuracy": None, "sum_posterior_accuracy": None, "validation_metric": "sum_posterior_accuracy"}
            for m in milestones
        }
        recent_losses: Deque[float] = deque(maxlen=max(validation_interval, 1))
        recent_masses: Deque[float] = deque(maxlen=max(validation_interval, 1))
        recent_zeros: Deque[int] = deque(maxlen=max(validation_interval, 1))
        started_at = time.perf_counter()

        validation_request_queue = None
        validation_result_queue = None
        validation_process = None
        pending_validation_snapshots: Dict[int, Dict[str, Any]] = {}
        submitted_validation_jobs = 0
        completed_validation_jobs = 0
        skipped_validation_jobs = 0
        async_stop_step: Optional[int] = None

        if async_enabled:
            ctx = mp.get_context("spawn")
            validation_request_queue = ctx.Queue(maxsize=async_max_pending)
            validation_result_queue = ctx.Queue()
            validation_process = ctx.Process(
                target=_validation_worker_main,
                kwargs={
                    "request_queue": validation_request_queue,
                    "result_queue": validation_result_queue,
                    "config": config,
                    "split_manifest": split_manifest,
                    "seed": seed,
                    "n_terms": n_terms,
                    "mode": mode,
                    "posterior_cases": posterior_cases,
                    "zero_total_mass_tolerance": zero_total_mass_tolerance,
                    "device_name": async_device_name,
                },
                daemon=True,
            )
            validation_process.start()

        def record_validation_result(
            result: Dict[str, Any],
            *,
            source: str,
            checkpoint_snapshot: Optional[Dict[str, Any]] = None,
            trainer_step_when_recorded: Optional[int] = None,
        ) -> bool:
            if result.get("type") == "error":
                raise RuntimeError(
                    f"Validation worker failed for step {result.get('step')}: "
                    f"{result.get('error_type')}: {result.get('error')}"
                )
            step = int(result["step"])
            elapsed = float(result["elapsed_seconds"])
            acc = float(result["sum_posterior_accuracy"] if result.get("sum_posterior_accuracy") is not None else result["digit_accuracy"])
            snapshot_step = int(result.get("validation_snapshot_step", step))
            trainer_step_submitted = result.get("trainer_step_when_submitted")
            if trainer_step_submitted is not None:
                trainer_step_submitted = int(trainer_step_submitted)
            validation_lag_steps = (
                int(trainer_step_when_recorded) - snapshot_step
                if trainer_step_when_recorded is not None
                else None
            )
            val_writer.writerow(
                {
                    "step": step,
                    "elapsed_seconds": elapsed,
                    # Backward-compatible alias for existing visualizers.
                    "digit_accuracy": acc,
                    "sum_posterior_accuracy": acc,
                    "sum_posterior_cases": result.get("sum_posterior_cases"),
                    "sum_posterior_candidate_count": result.get("sum_posterior_candidate_count"),
                    "sum_posterior_mean_true_mass": result.get("sum_posterior_mean_true_mass"),
                    "sum_posterior_mean_pred_mass": result.get("sum_posterior_mean_pred_mass"),
                    "sum_posterior_mean_total_mass": result.get("sum_posterior_mean_total_mass"),
                    "sum_posterior_zero_total_rate": result.get("sum_posterior_zero_total_rate"),
                    "sum_posterior_tie_rate": result.get("sum_posterior_tie_rate"),
                    "sum_posterior_branch_count_mean": result.get("sum_posterior_branch_count_mean"),
                    "validation_metric": "sum_posterior_accuracy",
                    "loss_recent_mean": result.get("loss_recent_mean"),
                    "true_mass_recent_mean": result.get("true_mass_recent_mean"),
                    "zero_true_mass_recent_rate": result.get("zero_true_mass_recent_rate"),
                    "validation_source": source,
                    "trainer_step_when_recorded": trainer_step_when_recorded,
                    "validation_snapshot_step": snapshot_step,
                    "trainer_step_when_submitted": trainer_step_submitted,
                    "validation_lag_steps": validation_lag_steps,
                }
            )
            val_handle.flush()
            reached_new = False
            for milestone in milestones:
                key = f"{milestone:.2f}"
                if not milestones_state[key]["reached"] and acc >= milestone:
                    milestones_state[key] = {
                        "reached": True,
                        "step": int(step),
                        "elapsed_seconds": float(elapsed),
                        "digit_accuracy": float(acc),
                        "sum_posterior_accuracy": float(acc),
                        "validation_metric": "sum_posterior_accuracy",
                    }
                    checkpoint_path = this_run_dir / "checkpoints" / f"milestone_{str(milestone).replace('.', 'p')}.pt"
                    if checkpoint_snapshot is None:
                        save_training_checkpoint(
                            checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            step=step,
                            elapsed_seconds=elapsed,
                            digit_accuracy_value=acc,
                            config=config,
                            seed=seed,
                            n_terms=n_terms,
                            mode=mode,
                            validation_metric_name="sum_posterior_accuracy",
                            sum_posterior_accuracy=acc,
                        )
                    else:
                        _save_training_checkpoint_snapshot(
                            checkpoint_path,
                            state_dict=checkpoint_snapshot["model_state_dict"],
                            optimizer_state_dict=checkpoint_snapshot["optimizer_state_dict"],
                            step=step,
                            elapsed_seconds=elapsed,
                            digit_accuracy_value=acc,
                            config=config,
                            seed=seed,
                            n_terms=n_terms,
                            mode=mode,
                            validation_metric_name="sum_posterior_accuracy",
                            sum_posterior_accuracy=acc,
                        )
                    reached_new = True
            if reached_new:
                _write_milestones_json(this_run_dir / "milestones.json", milestones_state)
            return bool(stop_at_highest and acc >= highest_milestone)

        def sync_validate_and_checkpoint(step: int) -> bool:
            elapsed = time.perf_counter() - started_at
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
            acc = float(metrics["sum_posterior_accuracy"])
            setattr(module, "readMNist", read_mnist)
            return record_validation_result(
                {
                    "step": int(step),
                    "elapsed_seconds": float(elapsed),
                    "digit_accuracy": float(acc),
                    **metrics,
                    "loss_recent_mean": recent_mean(list(recent_losses)),
                    "true_mass_recent_mean": recent_mean(list(recent_masses)),
                    "zero_true_mass_recent_rate": recent_mean(list(recent_zeros)),
                },
                source="sync",
                trainer_step_when_recorded=step,
            )

        def submit_async_validation(step: int, trainer_step_when_submitted: int) -> bool:
            nonlocal submitted_validation_jobs, skipped_validation_jobs
            if not async_enabled or validation_request_queue is None:
                return sync_validate_and_checkpoint(step)
            poll_async_validation_results(trainer_step_when_recorded=trainer_step_when_submitted)
            if len(pending_validation_snapshots) >= async_max_pending:
                skipped_validation_jobs += 1
                return False
            submitted_validation_jobs += 1
            job_id = submitted_validation_jobs
            snapshot = {
                "model_state_dict": _snapshot_model_state(model),
                "optimizer_state_dict": _snapshot_optimizer_state(optimizer),
                "step": int(step),
                "trainer_step_when_submitted": int(trainer_step_when_submitted),
            }
            request = {
                "type": "validate",
                "job_id": job_id,
                "step": int(step),
                "validation_snapshot_step": int(step),
                "trainer_step_when_submitted": int(trainer_step_when_submitted),
                "elapsed_seconds": float(time.perf_counter() - started_at),
                "model_state_dict": snapshot["model_state_dict"],
                "loss_recent_mean": recent_mean(list(recent_losses)),
                "true_mass_recent_mean": recent_mean(list(recent_masses)),
                "zero_true_mass_recent_rate": recent_mean(list(recent_zeros)),
            }
            try:
                validation_request_queue.put_nowait(request)
            except queue.Full:
                skipped_validation_jobs += 1
                return False
            pending_validation_snapshots[job_id] = snapshot
            return False

        def poll_async_validation_results(*, trainer_step_when_recorded: Optional[int]) -> bool:
            nonlocal completed_validation_jobs
            if not async_enabled or validation_result_queue is None:
                return False
            should_stop_now = False
            while True:
                try:
                    result = validation_result_queue.get_nowait()
                except queue.Empty:
                    break
                completed_validation_jobs += 1
                job_id = int(result.get("job_id", -1))
                snapshot = pending_validation_snapshots.pop(job_id, None)
                if record_validation_result(
                    result,
                    source=str(result.get("validation_source", "async_worker")),
                    checkpoint_snapshot=snapshot,
                    trainer_step_when_recorded=trainer_step_when_recorded,
                ):
                    should_stop_now = True
            return should_stop_now

        def shutdown_validation_worker() -> None:
            if validation_process is None or validation_request_queue is None:
                return
            try:
                validation_request_queue.put(
                    {"type": "shutdown"},
                    timeout=max(0.1, async_join_timeout_sec),
                )
            except Exception:
                pass
            validation_process.join(timeout=async_join_timeout_sec)
            if validation_process.is_alive():
                validation_process.terminate()
                validation_process.join(timeout=async_join_timeout_sec)

        should_stop = sync_validate_and_checkpoint(0) if (not async_enabled or async_validate_step_zero) else submit_async_validation(0, 0)
        progress = TerminalProgressBar(
            max_steps,
            desc=f"Train terms={n_terms} mode={mode_name} seed={seed}",
            unit="cases",
            enabled=bool(config.get("show_progress", True)),
        )
        cases_seen = 0
        optimizer_update = 0
        next_validation_step = validation_interval
        if should_stop:
            progress.finish(postfix="highest milestone already reached at step 0")
        else:
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
                loss_value = float(batch_stats["loss"])
                p_value = float(batch_stats["true_mass"])
                zero_rate = float(batch_stats["zero_true_mass"])
                recent_losses.extend(float(v) for v in batch_stats["case_loss_values"])
                recent_masses.extend(float(v) for v in batch_stats["case_true_mass_values"])
                recent_zeros.extend(int(v) for v in batch_stats["case_zero_flags"])
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
                        "loss": loss_value,
                        "true_mass": p_value,
                        "zero_true_mass": zero_rate,
                        "branch_count": branch_count_mean,
                        "branch_count_mean": branch_count_mean,
                        "branch_count_total": branch_count_total,
                        "grad_norm": batch_stats["grad_norm"],
                    }
                )
                if optimizer_update % 25 == 0 or cases_seen >= max_steps:
                    train_handle.flush()
                progress.update(batch_stats["batch_size"], postfix=f"loss={loss_value:.4g}, p={p_value:.4g}")

                if async_enabled and poll_async_validation_results(trainer_step_when_recorded=cases_seen):
                    async_stop_step = cases_seen
                    progress.finish(postfix=f"async validation reached highest milestone; stopped at case {cases_seen}")
                    break

                if cases_seen >= next_validation_step:
                    if submit_async_validation(cases_seen, cases_seen):
                        progress.finish(postfix=f"reached highest milestone at step {cases_seen}")
                        break
                    while next_validation_step <= cases_seen:
                        next_validation_step += validation_interval
            else:
                progress.finish(postfix="max_steps reached")

        if async_enabled:
            if poll_async_validation_results(trainer_step_when_recorded=cases_seen) and async_stop_step is None:
                async_stop_step = cases_seen
            shutdown_validation_worker()
            if poll_async_validation_results(trainer_step_when_recorded=cases_seen) and async_stop_step is None:
                async_stop_step = cases_seen

        final_elapsed = time.perf_counter() - started_at
        final_metrics = _evaluate_sum_posterior_validation(
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
        setattr(module, "readMNist", read_mnist)
        final_acc = float(final_metrics["sum_posterior_accuracy"])
        save_training_checkpoint(
            this_run_dir / "checkpoints" / "final.pt",
            model=model,
            optimizer=optimizer,
            step=cases_seen,
            elapsed_seconds=final_elapsed,
            digit_accuracy_value=final_acc,
            config=config,
            seed=seed,
            n_terms=n_terms,
            mode=mode,
            validation_metric_name="sum_posterior_accuracy",
            sum_posterior_accuracy=final_acc,
        )
        _write_milestones_json(this_run_dir / "milestones.json", milestones_state)
        write_json(
            this_run_dir / "run_summary.json",
            {
                "seed": int(seed),
                "n_terms": n_terms,
                "mode_name": mode_name,
                "top_k_cutoff": mode.get("top_k_cutoff"),
                "mode_order_position": int(mode_order_position),
                "mode_order_offset": int(mode_order_offset),
                "run_order_index": None if run_order_index is None else int(run_order_index),
                "max_steps": max_steps,
                "sum_batch_size": sum_batch_size,
                "completed_training_cases": cases_seen,
                "optimizer_updates": optimizer_update,
                "validation_async_enabled": async_enabled,
                "validation_async_device": async_device_name if async_enabled else None,
                "validation_async_submitted_jobs": submitted_validation_jobs,
                "validation_async_completed_jobs": completed_validation_jobs,
                "validation_async_skipped_jobs": skipped_validation_jobs,
                "async_stop_step": async_stop_step,
                "final_elapsed_seconds": final_elapsed,
                "final_digit_accuracy": final_acc,
                "final_sum_posterior_accuracy": final_acc,
                "final_sum_posterior_mean_true_mass": final_metrics.get("sum_posterior_mean_true_mass"),
                "final_sum_posterior_mean_total_mass": final_metrics.get("sum_posterior_mean_total_mass"),
                "final_sum_posterior_zero_total_rate": final_metrics.get("sum_posterior_zero_total_rate"),
                "validation_metric": "sum_posterior_accuracy",
                "milestones": milestones_state,
            },
        )
        train_handle.close()
        val_handle.close()
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
    stage_message(1, 4, "Writing resolved Pipeline II config snapshot")
    save_config(config, paths.config_used_path)

    stage_message(2, 4, "Verifying prepared manifests, checkpoints, and compiled SPLL artifacts")
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

    stage_message(3, 4, "Loading MNIST train partition")
    dataset = load_mnist_train_dataset(config)

    stage_message(4, 4, "Training through generated SPLL artifacts")
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train MNIST through generated SPLL artifacts for Pipeline II.")
    parser.add_argument("--config", required=True, help="Path to the Pipeline II YAML config.")
    args = parser.parse_args()
    run_train_stage(load_config(args.config))


if __name__ == "__main__":
    main()
