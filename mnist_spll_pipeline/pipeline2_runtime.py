from __future__ import annotations

import csv
import inspect
import math
import numbers
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import torch
import torch.nn as nn

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


class TensorLRUCache:
    def __init__(
        self,
        dataset: Any,
        *,
        device: torch.device,
        cache_device: str,
        max_items: int,
        strategy: str,
    ) -> None:
        self.dataset = dataset
        self.device = device
        self.cache_device = cache_device
        self.max_items = max(0, int(max_items))
        self.strategy = str(strategy).lower()
        if self.strategy not in {"none", "lru"}:
            raise ValueError("data.image_cache_strategy currently supports 'none' or 'lru'.")
        if self.cache_device not in {"cpu", "device"}:
            raise ValueError("data.image_cache_device must be 'cpu' or 'device'.")
        self._items: OrderedDict[int, torch.Tensor] = OrderedDict()

    def get(self, index: int) -> torch.Tensor:
        index = int(index)
        if self.strategy == "lru" and index in self._items:
            tensor = self._items.pop(index)
            self._items[index] = tensor
            return tensor.to(self.device) if tensor.device != self.device else tensor

        tensor, _ = self.dataset[index]
        target_device = self.device if self.cache_device == "device" else torch.device("cpu")
        tensor = tensor.to(target_device)
        if self.strategy == "lru" and self.max_items > 0:
            self._items[index] = tensor
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
        return tensor.to(self.device) if tensor.device != self.device else tensor

    def clear(self) -> None:
        self._items.clear()


class DifferentiableReadMNIST:
    def __init__(self, model: nn.Module, tensor_cache: TensorLRUCache, device: torch.device) -> None:
        self.model = model
        self.tensor_cache = tensor_cache
        self.device = device

    def __call__(self, global_index: int) -> torch.Tensor:
        inputs = self.tensor_cache.get(int(global_index)).unsqueeze(0).to(self.device)
        return torch.softmax(self.model(inputs), dim=-1)[0]


class PrecomputedReadMNIST:
    """Differentiable per-batch probability lookup installed as generated readMNist."""

    def __init__(self, probabilities_by_index: Dict[int, torch.Tensor]) -> None:
        self.probabilities_by_index = {
            int(index): probability for index, probability in probabilities_by_index.items()
        }
        self.call_count = 0

    def __call__(self, global_index: int) -> torch.Tensor:
        index = int(global_index)
        self.call_count += 1
        try:
            return self.probabilities_by_index[index]
        except KeyError as exc:
            raise KeyError(
                f"SPLL requested MNIST index {index}, but it was not precomputed for this batch."
            ) from exc


def make_precomputed_read_mnist(
    *,
    model: nn.Module,
    tensor_cache: TensorLRUCache,
    device: torch.device,
    global_indices: Sequence[int],
) -> PrecomputedReadMNIST:
    ordered_unique_indices = list(dict.fromkeys(int(value) for value in global_indices))
    if not ordered_unique_indices:
        raise ValueError("Cannot build a precomputed readMNist lookup for an empty batch.")
    inputs = torch.stack(
        [tensor_cache.get(index).to(device) for index in ordered_unique_indices],
        dim=0,
    )
    probabilities = torch.softmax(model(inputs), dim=-1)
    return PrecomputedReadMNIST(
        {
            index: probabilities[row_index]
            for row_index, index in enumerate(ordered_unique_indices)
        }
    )


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
            "adaptive_top_k": bool(mode.get("adaptive_top_k", False)),
            "posterior_mass_target": mode.get("posterior_mass_target"),
            "cutoff_search": mode.get("cutoff_search"),
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
    cache: TensorLRUCache,
    device: torch.device,
    cases: list[Dict[str, Any]],
    loss_epsilon: float,
) -> Dict[str, Any]:
    """Apply one optimizer update over a batch of generated SPLL sum cases."""

    if not cases:
        raise ValueError("Cannot train on an empty SPLL sum batch.")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    all_indices = [
        int(index)
        for case in cases
        for index in case["global_indices"]
    ]
    batch_read_mnist = make_precomputed_read_mnist(
        model=model,
        tensor_cache=cache,
        device=device,
        global_indices=all_indices,
    )
    setattr(module, "readMNist", batch_read_mnist)

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
    read_mnist_lookup_total = int(getattr(batch_read_mnist, "call_count", 0))
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
        "read_mnist_lookup_mean": (read_mnist_lookup_total / len(cases)) if cases else None,
        "read_mnist_lookup_total": read_mnist_lookup_total,
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
