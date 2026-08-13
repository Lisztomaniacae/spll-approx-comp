from __future__ import annotations

import csv
import inspect
import math
import numbers
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from pipeline2_config import PIPELINE2_READ_MNIST_POLICY
from pipeline_support import ensure_dir, utc_now_iso, write_json
from spll_artifacts import _get_tuple_item, extract_branch_count


def _is_numeric_zero(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, numbers.Real) and float(value) == 0.0


def _is_zero_tensor(value: torch.Tensor) -> bool:
    return value.numel() == 1 and float(value.detach().cpu().item()) == 0.0


def _pruned_zero_tensor(zero_anchor: Optional[torch.Tensor]) -> torch.Tensor:
    return zero_anchor * 0.0 if zero_anchor is not None else torch.tensor(0.0, dtype=torch.float32)


def zero_anchor_from_model(model: nn.Module) -> torch.Tensor:
    for parameter in model.parameters():
        if parameter.requires_grad:
            return parameter.sum() * 0.0
    raise RuntimeError("Cannot build a differentiable zero anchor: model has no trainable parameters.")


def _as_probability_tensor(
    return_value: Any,
    *,
    allow_pruned_zero: bool = False,
    zero_anchor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(return_value, torch.Tensor):
        if (
            allow_pruned_zero
            and zero_anchor is not None
            and not return_value.requires_grad
            and _is_zero_tensor(return_value)
        ):
            return _pruned_zero_tensor(zero_anchor)
        return return_value
    if allow_pruned_zero and _is_numeric_zero(return_value):
        return _pruned_zero_tensor(zero_anchor)

    probability = _get_tuple_item(return_value, 0)
    if isinstance(probability, torch.Tensor):
        if (
            allow_pruned_zero
            and zero_anchor is not None
            and not probability.requires_grad
            and _is_zero_tensor(probability)
        ):
            return _pruned_zero_tensor(zero_anchor)
        return probability
    if allow_pruned_zero and _is_numeric_zero(probability):
        return _pruned_zero_tensor(zero_anchor)

    raise TypeError(
        "Generated SPLL artifact returned a non-tensor probability. "
        "Pipeline II requires a differentiable torch.Tensor probability, except for literal zero "
        "returned by a fully pruned true-sum path."
    )


def call_true_sum(
    module: Any,
    true_sum: int,
    indices: Sequence[int],
    *,
    allow_pruned_zero: bool = False,
    zero_anchor: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[int]]:
    forward = module.main.forward
    arguments = [int(value) for value in indices]
    result = (
        forward(int(true_sum), 1.0, *arguments)
        if "acc_prob" in inspect.signature(forward).parameters
        else forward(int(true_sum), *arguments)
    )
    probability = _as_probability_tensor(
        result,
        allow_pruned_zero=allow_pruned_zero,
        zero_anchor=zero_anchor,
    )
    return probability, extract_branch_count(result)


class DirectReadMNIST:
    """Generated ``readMNist`` callback for Pipeline II training.

    Every generated callback invocation reloads the requested MNIST tensor from
    the dataset and executes a fresh CNN forward pass.  No image tensor cache,
    probability lookup, memoization, or cross-call reuse is permitted in this
    benchmark path.  The returned softmax row remains attached to autograd.
    """

    def __init__(self, model: nn.Module, dataset: Any, device: torch.device) -> None:
        self.model = model
        self.dataset = dataset
        self.device = device
        self.call_count = 0
        self.model_evaluation_count = 0
        self.unique_indices_seen: set[int] = set()

    def reset_counts(self) -> None:
        self.call_count = 0
        self.model_evaluation_count = 0
        self.unique_indices_seen.clear()

    def __call__(self, global_index: int) -> torch.Tensor:
        index = int(global_index)
        self.call_count += 1
        self.unique_indices_seen.add(index)
        inputs, _ = self.dataset[index]
        inputs = inputs.unsqueeze(0).to(self.device)
        self.model_evaluation_count += 1
        return torch.softmax(self.model(inputs), dim=-1)[0]

    def stats(self) -> Dict[str, Any]:
        return {
            "policy": PIPELINE2_READ_MNIST_POLICY,
            "calls": int(self.call_count),
            "model_evaluations": int(self.model_evaluation_count),
            "unique_indices": int(len(self.unique_indices_seen)),
        }


def get_runtime_top_k_cutoff(module: Any, mode: Dict[str, Any]) -> Optional[float]:
    """Return the fixed cutoff actually exposed by the generated artifact."""

    configured = mode.get("top_k_cutoff")
    if configured is None:
        return None
    runtime = getattr(module, "TOP_K_CUTOFF", configured)
    return float(runtime)


def tensor_to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def validate_probability_and_loss(
    p_true: torch.Tensor,
    loss: torch.Tensor,
    *,
    step: int,
) -> None:
    probability = tensor_to_float(p_true)
    loss_value = tensor_to_float(loss)
    if math.isnan(probability) or math.isinf(probability):
        raise FloatingPointError(f"Invalid true-sum probability at step {step}: {probability}")
    if probability < 0.0:
        raise FloatingPointError(f"Negative true-sum probability at step {step}: {probability}")
    if math.isnan(loss_value) or math.isinf(loss_value):
        raise FloatingPointError(f"Invalid loss at step {step}: {loss_value}")


def grad_norm(model: nn.Module) -> float:
    total_squared_norm = 0.0
    has_gradient = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Encountered NaN/inf gradient.")
        norm = float(gradient.norm().detach().cpu().item())
        total_squared_norm += norm * norm
        has_gradient = True
    return math.sqrt(total_squared_norm) if has_gradient else 0.0


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    elapsed_seconds: float,
    digit_accuracy_value: float,
    config: Dict[str, Any],
    seed: int,
    n_terms: int,
    mode: Dict[str, Any],
    validation_metric_name: str = "digit_accuracy",
    sum_posterior_accuracy: Optional[float] = None,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": int(step),
            "elapsed_seconds": float(elapsed_seconds),
            "digit_accuracy": float(digit_accuracy_value),
            "validation_metric_name": str(validation_metric_name),
            "sum_posterior_accuracy": (
                float(sum_posterior_accuracy) if sum_posterior_accuracy is not None else None
            ),
            "model_config": dict(config.get("model", {})),
            "seed": int(seed),
            "n_terms": int(n_terms),
            "inference_mode": mode["name"],
            "artifact_name": mode.get("artifact_name", mode["name"]),
            "top_k_cutoff": mode.get("top_k_cutoff"),
            "read_mnist_policy": PIPELINE2_READ_MNIST_POLICY,
            "config_path": str(config.get("_config_path", "")),
            "created_at_utc": utc_now_iso(),
        },
        path,
    )


def write_failure_report(
    path: Path,
    *,
    error: BaseException,
    context: Dict[str, Any],
) -> None:
    write_json(
        path,
        {
            "created_at_utc": utc_now_iso(),
            "error_type": type(error).__name__,
            "error": str(error),
            "context": context,
        },
    )


def cleanup_torch() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


@contextmanager
def open_csv_trace(
    path: Path,
    fieldnames: Sequence[str],
) -> Iterator[Tuple[Any, csv.DictWriter]]:
    """Open a CSV trace and yield both its handle and initialized writer."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        yield handle, writer


def recent_mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def train_sum_batch(
    *,
    module: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    read_mnist: DirectReadMNIST,
    cases: list[Dict[str, Any]],
    loss_epsilon: float,
) -> Dict[str, Any]:
    """Apply one optimizer update over a batch of generated SPLL sum cases."""

    if not cases:
        raise ValueError("Cannot train on an empty SPLL sum batch.")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if read_mnist.model is not model:
        raise RuntimeError("DirectReadMNIST is bound to a different model than train_sum_batch().")
    read_mnist.reset_counts()
    setattr(module, "readMNist", read_mnist)

    zero_anchor = zero_anchor_from_model(model)
    probabilities = []
    branch_counts = []
    for case in cases:
        probability, branch_count = call_true_sum(
            module,
            int(case["true_sum"]),
            case["global_indices"],
            allow_pruned_zero=True,
            zero_anchor=zero_anchor,
        )
        if not isinstance(probability, torch.Tensor) or not probability.requires_grad:
            raise RuntimeError(
                f"Detached/non-tensor probability returned at training case {case['step']}."
            )
        probabilities.append(probability)
        branch_counts.append(branch_count)

    probability_batch = torch.stack(probabilities)
    per_case_losses = -torch.log(probability_batch + loss_epsilon)
    loss = per_case_losses.mean()
    for case, probability, case_loss in zip(cases, probabilities, per_case_losses):
        validate_probability_and_loss(probability, case_loss, step=int(case["step"]))
    validate_probability_and_loss(
        probability_batch.mean(),
        loss,
        step=int(cases[-1]["step"]),
    )

    loss.backward()
    gradient_norm = grad_norm(model)
    optimizer.step()

    probability_values = [tensor_to_float(value) for value in probabilities]
    loss_values = [tensor_to_float(value) for value in per_case_losses]
    zero_flags = [int(value <= 0.0) for value in probability_values]
    numeric_branch_counts = [int(value) for value in branch_counts if value is not None]
    read_mnist_calls = int(read_mnist.call_count)
    read_mnist_model_evaluations = int(read_mnist.model_evaluation_count)
    if read_mnist_calls != read_mnist_model_evaluations:
        raise RuntimeError(
            "Pipeline II direct readMNist invariant violated: every generated readMNist call "
            "must execute exactly one MNIST model forward."
        )
    return {
        "batch_size": len(cases),
        "loss": float(sum(loss_values) / len(loss_values)),
        "true_mass": float(sum(probability_values) / len(probability_values)),
        "zero_true_mass": float(sum(zero_flags) / len(zero_flags)),
        "branch_count_mean": (
            float(sum(numeric_branch_counts) / len(numeric_branch_counts))
            if numeric_branch_counts
            else None
        ),
        "branch_count_total": (
            int(sum(numeric_branch_counts)) if numeric_branch_counts else None
        ),
        "read_mnist_calls_mean": (read_mnist_calls / len(cases)) if cases else None,
        "read_mnist_calls_total": read_mnist_calls,
        "read_mnist_model_evaluations_mean": (
            read_mnist_model_evaluations / len(cases)
        ) if cases else None,
        "read_mnist_model_evaluations_total": read_mnist_model_evaluations,
        "grad_norm": gradient_norm,
        "case_loss_values": loss_values,
        "case_true_mass_values": probability_values,
        "case_zero_flags": zero_flags,
    }


def load_initial_model(
    config: Dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    from pipeline2_data import build_model_from_config

    payload = torch.load(checkpoint_path, map_location="cpu")
    model = build_model_from_config(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def make_optimizer(
    config: Dict[str, Any],
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    optimizer_cfg = config.get("optimizer", {})
    return torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_cfg.get("learning_rate", 0.001)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )


def run_preflight(
    *,
    module: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    split_manifest: Dict[str, Any],
    seed: int,
    n_terms: int,
    loss_epsilon: float,
) -> None:
    from pipeline2_data import generate_sum_case

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
        probability, _ = call_true_sum(
            module,
            int(case["true_sum"]),
            case["global_indices"],
            allow_pruned_zero=True,
            zero_anchor=zero_anchor_from_model(model),
        )
        if not isinstance(probability, torch.Tensor) or not probability.requires_grad:
            raise RuntimeError(
                "Generated SPLL artifact returned a probability that is not a differentiable torch.Tensor."
            )
        loss = -torch.log(probability + loss_epsilon)
        validate_probability_and_loss(probability, loss, step=0)
        loss.backward()
        if grad_norm(model) > 0.0:
            optimizer.zero_grad(set_to_none=True)
            return
        last_zero_mass = tensor_to_float(probability) <= 0.0

    reason = (
        "only fully pruned zero-mass cases were observed"
        if last_zero_mass
        else "no nonzero gradient was produced"
    )
    raise RuntimeError(
        f"Preflight differentiability check failed after {max_probe_steps} probe cases: {reason}."
    )
