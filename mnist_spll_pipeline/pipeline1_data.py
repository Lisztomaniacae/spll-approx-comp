from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import ConcatDataset

from mnist_model import build_mnist_transform, load_checkpoint_model, load_full_mnist_raw
from pipeline1_config import PipelinePaths
from pipeline1_models import default_rng
from pipeline_support import TerminalProgressBar, ensure_dir, load_json, load_config, resolve_path
from spll_artifacts import extract_branch_count, extract_probability


def build_read_mnist(model_path: Path, device: torch.device, config_path: Path):
    config = load_config(config_path)
    model = load_checkpoint_model(model_path, config, map_location="cpu")
    model.to(device)
    model.eval()
    transform = build_mnist_transform(config)

    from PIL import Image

    def read_mnist(image_path: str) -> List[float]:
        # Intentionally uncached. The inference engine installs a fresh cache or
        # lookup proxy for exactly one measured query when configured to do so.
        image = Image.open(image_path).convert("L")
        inputs = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(inputs), dim=-1)[0].detach().cpu().tolist()

        values = [float(value) for value in probabilities]
        if len(values) != 10 or any(not math.isfinite(value) for value in values):
            raise ValueError(f"readMNist returned invalid probabilities for {image_path!r}: {values!r}")
        return values

    return read_mnist


def sample_experiments(
    raw_dataset: ConcatDataset,
    inference_indices: Sequence[int],
    num_experiments: int,
    terms_min: int,
    terms_max: int,
    without_replacement_within_experiment: bool,
    rng,
    inputs_root: Path,
    *,
    show_progress: bool,
) -> List[Dict[str, Any]]:
    ensure_dir(inputs_root)
    if terms_max > len(inference_indices):
        raise ValueError(
            f"terms_per_sum_max={terms_max} exceeds the inference subset size {len(inference_indices)}."
        )

    experiments: List[Dict[str, Any]] = []
    term_counts = rng.integers(low=terms_min, high=terms_max + 1, size=num_experiments)
    progress = TerminalProgressBar(
        num_experiments,
        desc="Staging",
        unit="experiments",
        enabled=show_progress and num_experiments > 0,
    )

    for experiment_id, n_terms in enumerate(term_counts, start=1):
        chosen_positions = rng.choice(
            len(inference_indices),
            size=int(n_terms),
            replace=not without_replacement_within_experiment,
        )
        chosen_global_indices = [int(inference_indices[position]) for position in chosen_positions]

        experiment_dir = ensure_dir(inputs_root / f"experiment_{experiment_id:04d}")
        image_paths: List[str] = []
        labels: List[int] = []
        for local_index, global_index in enumerate(chosen_global_indices):
            image, label = raw_dataset[global_index]
            image_path = experiment_dir / (
                f"term_{local_index:02d}_global_{global_index:05d}_label_{label}.png"
            )
            image.save(image_path)
            image_paths.append(str(image_path.resolve()))
            labels.append(int(label))

        experiments.append(
            {
                "experiment_id": experiment_id,
                "n_terms": int(n_terms),
                "global_indices": chosen_global_indices,
                "image_paths": image_paths,
                "labels": labels,
                "true_sum": int(sum(labels)),
            }
        )
        progress.update(postfix=f"exp={experiment_id:04d}, terms={int(n_terms)}")

    progress.finish(postfix="all staged experiments ready")
    return experiments


def load_split_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    path = resolve_path(config, config["paths"]["split_manifest"])
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found at {path}. Run train_mnist.py first.")
    return torch.load(path, map_location="cpu")


def stage_experiment_bundle(
    config: Dict[str, Any],
    *,
    inference_cfg: Dict[str, Any],
    inputs_root: Path,
    show_progress: bool,
    terms_min: int,
    terms_max: int,
) -> List[Dict[str, Any]]:
    split_manifest = load_split_manifest(config)
    inference_indices = list(split_manifest["inference_indices"])
    raw_dataset = load_full_mnist_raw(config)
    return sample_experiments(
        raw_dataset=raw_dataset,
        inference_indices=inference_indices,
        num_experiments=int(inference_cfg.get("num_experiments", 200)),
        terms_min=terms_min,
        terms_max=terms_max,
        without_replacement_within_experiment=bool(
            inference_cfg.get("sample_without_replacement_within_experiment", True)
        ),
        rng=default_rng(config, offset=100),
        inputs_root=inputs_root,
        show_progress=show_progress,
    )


def load_staged_experiments(paths: PipelinePaths) -> List[Dict[str, Any]]:
    if not paths.staged_experiments_path.exists():
        raise FileNotFoundError(
            f"Staged experiments not found at {paths.staged_experiments_path}. Run the 'stage' step first."
        )
    payload = load_json(paths.staged_experiments_path)
    experiments = payload.get("experiments") if isinstance(payload, dict) else payload
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"Expected a non-empty experiment list in {paths.staged_experiments_path}.")
    return experiments


def evaluate_candidate_sum(
    module,
    image_paths,
    candidate_sum: int,
    *,
    expects_acc_prob: Optional[bool] = None,
) -> Dict[str, Any]:
    if expects_acc_prob is None:
        expects_acc_prob = "acc_prob" in inspect.signature(module.main.forward).parameters

    candidate = int(candidate_sum)
    result = (
        module.main.forward(candidate, 1.0, *image_paths)
        if expects_acc_prob
        else module.main.forward(candidate, *image_paths)
    )
    return {
        "candidate_sum": candidate,
        "probability_raw": float(extract_probability(result)),
        "branch_count": extract_branch_count(result),
    }


def posterior_for_experiment(
    module,
    image_paths,
    max_sum,
    *,
    progress_bar=None,
    progress_prefix="",
):
    posterior = []
    branch_counts = []
    expects_acc_prob = "acc_prob" in inspect.signature(module.main.forward).parameters

    for candidate in range(max_sum + 1):
        trace = evaluate_candidate_sum(
            module,
            image_paths,
            candidate,
            expects_acc_prob=expects_acc_prob,
        )
        posterior.append(trace["probability_raw"])
        branch_counts.append(trace["branch_count"])
        if progress_bar is not None:
            progress_bar.update(postfix=f"{progress_prefix} sum={candidate}")

    return {"posterior_raw": posterior, "branch_counts_raw": branch_counts}
