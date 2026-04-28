from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import torch

from mnist_spll_common import TerminalProgressBar, load_config, save_config, set_seed, stage_message
from mnist_spll_pipeline_core import load_json, write_json
from spll_training_core import (
    DifferentiableReadMNIST,
    TensorLRUCache,
    assert_compiled_artifacts_exist,
    build_model_from_config,
    call_true_sum,
    cleanup_torch,
    compiled_program_path,
    digit_accuracy,
    extract_validation_indices,
    generate_sum_case,
    get_experiments,
    get_inference_modes,
    get_seeds,
    grad_norm,
    initial_checkpoint_path,
    load_compiled_training_module,
    load_mnist_train_dataset,
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


def _write_milestones_json(path: Path, milestones_state: Dict[str, Dict[str, Any]]) -> None:
    write_json(path, {"milestones": milestones_state})


def _train_one_run(
    *,
    config: Dict[str, Any],
    split_manifest: Dict[str, Any],
    dataset: Any,
    validation_indices: List[int],
    seed: int,
    experiment: Dict[str, Any],
    mode: Dict[str, Any],
) -> None:
    paths = training_paths(config)
    n_terms = int(experiment["n_terms"])
    max_steps = int(experiment["max_steps"])
    mode_name = str(mode["name"])
    this_run_dir = run_dir(paths, seed, n_terms, mode_name)
    ensure_dir(this_run_dir)
    ensure_dir(this_run_dir / "checkpoints")

    context = {"seed": seed, "n_terms": n_terms, "mode": mode_name, "top_k_cutoff": mode.get("top_k_cutoff")}
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
        setattr(module, "readMNist", DifferentiableReadMNIST(model, cache, device))

        loss_epsilon = float(config.get("training", {}).get("loss_epsilon", 1.0e-12))
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
        validation_interval = int(validation_cfg.get("interval_steps", 100))
        eval_batch_size = int(validation_cfg.get("digit_eval_batch_size", 256))
        milestones = [float(v) for v in validation_cfg.get("milestones", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])]
        highest_milestone = max(milestones)
        stop_at_highest = bool(validation_cfg.get("stop_at_highest_milestone", True))
        sum_probe_cfg = validation_cfg.get("sum_probe", {})
        sum_probe_enabled = bool(sum_probe_cfg.get("enabled", True))
        sum_probe_cases = int(sum_probe_cfg.get("num_cases", 100)) if sum_probe_enabled else 0

        train_handle, train_writer = write_csv_header(
            this_run_dir / "train_trace.csv",
            [
                "step",
                "elapsed_seconds",
                "loss",
                "true_mass",
                "zero_true_mass",
                "branch_count",
                "grad_norm",
            ],
        )
        val_handle, val_writer = write_csv_header(
            this_run_dir / "validation_trace.csv",
            [
                "step",
                "elapsed_seconds",
                "digit_accuracy",
                "sum_probe_mean_true_mass",
                "loss_recent_mean",
                "true_mass_recent_mean",
                "zero_true_mass_recent_rate",
            ],
        )

        milestones_state: Dict[str, Dict[str, Any]] = {
            f"{m:.2f}": {"reached": False, "step": None, "elapsed_seconds": None, "digit_accuracy": None}
            for m in milestones
        }
        recent_losses: Deque[float] = deque(maxlen=max(validation_interval, 1))
        recent_masses: Deque[float] = deque(maxlen=max(validation_interval, 1))
        recent_zeros: Deque[int] = deque(maxlen=max(validation_interval, 1))
        started_at = time.perf_counter()

        def validate_and_checkpoint(step: int) -> bool:
            elapsed = time.perf_counter() - started_at
            acc = digit_accuracy(model, dataset, validation_indices, device, eval_batch_size)
            sum_probe = _evaluate_sum_probe(
                module=module,
                split_manifest=split_manifest,
                seed=seed,
                n_terms=n_terms,
                num_cases=sum_probe_cases,
            )
            val_writer.writerow(
                {
                    "step": step,
                    "elapsed_seconds": elapsed,
                    "digit_accuracy": acc,
                    "sum_probe_mean_true_mass": sum_probe,
                    "loss_recent_mean": recent_mean(list(recent_losses)),
                    "true_mass_recent_mean": recent_mean(list(recent_masses)),
                    "zero_true_mass_recent_rate": recent_mean(list(recent_zeros)),
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
                    }
                    save_training_checkpoint(
                        this_run_dir / "checkpoints" / f"milestone_{str(milestone).replace('.', 'p')}.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        elapsed_seconds=elapsed,
                        digit_accuracy_value=acc,
                        config=config,
                        seed=seed,
                        n_terms=n_terms,
                        mode=mode,
                    )
                    reached_new = True
            if reached_new:
                _write_milestones_json(this_run_dir / "milestones.json", milestones_state)
            return bool(stop_at_highest and acc >= highest_milestone)

        should_stop = validate_and_checkpoint(0)
        progress = TerminalProgressBar(
            max_steps,
            desc=f"Train terms={n_terms} mode={mode_name} seed={seed}",
            unit="steps",
            enabled=bool(config.get("show_progress", True)),
        )
        if should_stop:
            progress.finish(postfix="highest milestone already reached at step 0")
        else:
            for step in range(1, max_steps + 1):
                model.train()
                case = generate_sum_case(
                    split_manifest,
                    base_seed=seed,
                    n_terms=n_terms,
                    step=step,
                    split="train",
                )
                optimizer.zero_grad(set_to_none=True)
                p_true, branch_count = call_true_sum(
                    module,
                    int(case["true_sum"]),
                    case["global_indices"],
                    allow_pruned_zero=True,
                    zero_anchor=zero_anchor_from_model(model),
                )
                if not isinstance(p_true, torch.Tensor) or not p_true.requires_grad:
                    raise RuntimeError(f"Detached/non-tensor probability returned at training step {step}.")
                loss = -torch.log(p_true + loss_epsilon)
                validate_probability_and_loss(p_true, loss, step=step)
                loss.backward()
                norm = grad_norm(model)
                optimizer.step()

                elapsed = time.perf_counter() - started_at
                p_value = tensor_to_float(p_true)
                loss_value = tensor_to_float(loss)
                zero_flag = int(p_value <= 0.0)
                recent_losses.append(loss_value)
                recent_masses.append(p_value)
                recent_zeros.append(zero_flag)
                train_writer.writerow(
                    {
                        "step": step,
                        "elapsed_seconds": elapsed,
                        "loss": loss_value,
                        "true_mass": p_value,
                        "zero_true_mass": zero_flag,
                        "branch_count": branch_count,
                        "grad_norm": norm,
                    }
                )
                if step % 25 == 0:
                    train_handle.flush()
                progress.update(postfix=f"loss={loss_value:.4g}, p={p_value:.4g}")

                if step % validation_interval == 0:
                    if validate_and_checkpoint(step):
                        progress.finish(postfix=f"reached highest milestone at step {step}")
                        break
            else:
                progress.finish(postfix="max_steps reached")

        final_elapsed = time.perf_counter() - started_at
        final_acc = digit_accuracy(model, dataset, validation_indices, device, eval_batch_size)
        save_training_checkpoint(
            this_run_dir / "checkpoints" / "final.pt",
            model=model,
            optimizer=optimizer,
            step=max([0] + [int(v.get("step") or 0) for v in milestones_state.values()]),
            elapsed_seconds=final_elapsed,
            digit_accuracy_value=final_acc,
            config=config,
            seed=seed,
            n_terms=n_terms,
            mode=mode,
        )
        _write_milestones_json(this_run_dir / "milestones.json", milestones_state)
        write_json(
            this_run_dir / "run_summary.json",
            {
                "seed": int(seed),
                "n_terms": n_terms,
                "mode_name": mode_name,
                "top_k_cutoff": mode.get("top_k_cutoff"),
                "max_steps": max_steps,
                "final_elapsed_seconds": final_elapsed,
                "final_digit_accuracy": final_acc,
                "milestones": milestones_state,
            },
        )
        train_handle.close()
        val_handle.close()
        cache.clear()
        cleanup_torch()
    except BaseException as exc:
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

    stage_message(3, 4, "Loading MNIST train partition and validation index set")
    dataset = load_mnist_train_dataset(config)
    validation_indices = extract_validation_indices(split_manifest)

    stage_message(4, 4, "Training through generated SPLL artifacts")
    total = len(get_seeds(config)) * len(get_experiments(config)) * len(get_inference_modes(config))
    outer = TerminalProgressBar(total, desc="Pipeline II runs", unit="runs", enabled=bool(config.get("show_progress", True)))
    for seed in get_seeds(config):
        for experiment in get_experiments(config):
            for mode in get_inference_modes(config):
                _train_one_run(
                    config=config,
                    split_manifest=split_manifest,
                    dataset=dataset,
                    validation_indices=validation_indices,
                    seed=seed,
                    experiment=experiment,
                    mode=mode,
                )
                outer.update(postfix=f"seed={seed}, terms={experiment['n_terms']}, mode={mode['name']}")
    outer.finish(postfix="done")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train MNIST through generated SPLL artifacts for Pipeline II.")
    parser.add_argument("--config", required=True, help="Path to the Pipeline II YAML config.")
    args = parser.parse_args()
    run_train_stage(load_config(args.config))


if __name__ == "__main__":
    main()
