from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch

from pipeline2_data import generate_sum_case
from pipeline2_runtime import (
    TensorLRUCache,
    call_true_sum,
    make_precomputed_read_mnist,
    tensor_to_float,
)


def is_adaptive_top_k_mode(mode: Dict[str, Any]) -> bool:
    return bool(mode.get("adaptive_top_k")) and mode.get("posterior_mass_target") is not None


def require_mutable_top_k_cutoff(module: Any, *, mode_name: str) -> None:
    if not hasattr(module, "TOP_K_CUTOFF"):
        raise RuntimeError(
            f"Adaptive top-k mode {mode_name!r} requires a generated SPLL artifact compiled with -k. "
            "The exact artifact does not expose TOP_K_CUTOFF. Re-run the Pipeline II compile stage."
        )


def set_runtime_top_k_cutoff(module: Any, cutoff: float) -> None:
    setattr(module, "TOP_K_CUTOFF", max(0.0, min(1.0, float(cutoff))))


def get_runtime_top_k_cutoff(module: Any, mode: Dict[str, Any]) -> Optional[float]:
    if hasattr(module, "TOP_K_CUTOFF"):
        try:
            return float(getattr(module, "TOP_K_CUTOFF"))
        except (TypeError, ValueError):
            return None
    cutoff = mode.get("top_k_cutoff")
    return None if cutoff is None else float(cutoff)


def candidate_sums_for_terms(n_terms: int) -> range:
    return range(0, 9 * int(n_terms) + 1)


def validate_probability_value(value: float, *, candidate_sum: int, case_step: int) -> None:
    if math.isnan(value) or math.isinf(value) or value < 0.0:
        raise FloatingPointError(
            f"Invalid posterior probability {value} for candidate_sum={candidate_sum}, "
            f"case_step={case_step}."
        )


def mean_surviving_posterior_mass_for_cutoff(
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
    """Measure mean raw posterior mass retained at one runtime cutoff."""

    previous_cutoff = getattr(module, "TOP_K_CUTOFF", None)
    previous_read_mnist = getattr(module, "readMNist", None)
    was_training = model.training
    set_runtime_top_k_cutoff(module, cutoff)
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
                for candidate_sum in candidate_sums_for_terms(n_terms):
                    probability, _ = call_true_sum(
                        module,
                        int(candidate_sum),
                        case["global_indices"],
                        allow_pruned_zero=True,
                    )
                    value = tensor_to_float(probability)
                    validate_probability_value(
                        value,
                        candidate_sum=int(candidate_sum),
                        case_step=int(case["step"]),
                    )
                    mass += value
                total_masses.append(mass)
    finally:
        if previous_cutoff is not None:
            setattr(module, "TOP_K_CUTOFF", previous_cutoff)
        if previous_read_mnist is not None:
            setattr(module, "readMNist", previous_read_mnist)
        if was_training:
            model.train()

    return float(sum(total_masses) / len(total_masses)) if total_masses else 0.0


def tune_adaptive_top_k_cutoff(
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
    del config  # Search parameters are already normalized into the mode.
    if not is_adaptive_top_k_mode(mode):
        return None
    require_mutable_top_k_cutoff(module, mode_name=str(mode["name"]))

    target = float(mode.get("posterior_mass_target", 0.8))
    search_cfg = dict(mode.get("cutoff_search") or {})
    probe_cases = max(1, int(search_cfg.get("probe_cases", 20)))
    max_iterations = max(1, int(search_cfg.get("max_iterations", 14)))
    tolerance = max(0.0, float(search_cfg.get("tolerance", 0.02)))
    low = max(0.0, min(1.0, float(search_cfg.get("min_cutoff", 0.0))))
    high = max(low, min(1.0, float(search_cfg.get("max_cutoff", 1.0))))

    def evaluate(cutoff: float) -> float:
        return mean_surviving_posterior_mass_for_cutoff(
            module=module,
            model=model,
            cache=cache,
            device=device,
            split_manifest=split_manifest,
            seed=seed,
            n_terms=n_terms,
            num_cases=probe_cases,
            cutoff=cutoff,
        )

    evaluations: List[Dict[str, float]] = []
    low_mass = evaluate(low)
    evaluations.append({"cutoff": low, "mean_surviving_mass": low_mass})
    best_cutoff = low
    best_mass = low_mass
    best_error = abs(best_mass - target)

    def result(iterations: int) -> Dict[str, Any]:
        set_runtime_top_k_cutoff(module, best_cutoff)
        return {
            "adaptive_top_k": True,
            "posterior_mass_target": target,
            "runtime_top_k_cutoff": float(best_cutoff),
            "mean_surviving_posterior_mass": float(best_mass),
            "abs_error": float(best_error),
            "probe_cases": probe_cases,
            "iterations": int(iterations),
            "converged": bool(best_error <= tolerance),
            "reason": str(reason),
            "step": int(step),
            "search_method": "bounded_monotone_bisection",
            "evaluations": evaluations,
        }

    if low_mass <= target or target >= 1.0:
        return result(0)

    high_mass = evaluate(high)
    evaluations.append({"cutoff": high, "mean_surviving_mass": high_mass})
    if abs(high_mass - target) < best_error:
        best_cutoff = high
        best_mass = high_mass
        best_error = abs(best_mass - target)
    if high_mass >= target:
        return result(0)

    left = low
    right = high
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        midpoint = (left + right) / 2.0
        midpoint_mass = evaluate(midpoint)
        evaluations.append(
            {"cutoff": midpoint, "mean_surviving_mass": midpoint_mass}
        )
        midpoint_error = abs(midpoint_mass - target)
        if midpoint_error < best_error:
            best_cutoff = midpoint
            best_mass = midpoint_mass
            best_error = midpoint_error
        if midpoint_error <= tolerance:
            break
        if midpoint_mass > target:
            left = midpoint
        else:
            right = midpoint

    return result(iterations)
