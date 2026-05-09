from __future__ import annotations

import argparse
import cProfile
import csv
import dis
import gc
import hashlib
import importlib.util
import inspect
import io
import json
import math
import platform
import pstats
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


VariantName = str
ReadMNistFn = Callable[[str], Sequence[float]]


def utc_now_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def threshold_label(cutoff: Optional[float]) -> str:
    if cutoff is None:
        return "exact"
    return f"cutoff_{str(cutoff).replace('.', 'p')}"


def compiled_program_path(outputs_root: Path, n_terms: int, cutoff_mode: str, cutoff: Optional[float]) -> Path:
    return (
        outputs_root
        / "spll_experiments"
        / "generated"
        / "compiled_python"
        / cutoff_mode
        / f"sum_{int(n_terms):02d}"
        / threshold_label(cutoff)
        / "program.py"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_compiled_module(module_path: Path, module_label: str) -> Any:
    module_path = module_path.resolve()
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    module_name = f"diagnose_cutoff_zero_{module_label}_{hashlib.sha1(str(module_path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import compiled SPLL module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_tuple_item(value: Any, index: int) -> Any:
    try:
        return value[index]
    except Exception:
        pass
    attr_name = "t1" if index == 0 else "t2"
    if hasattr(value, attr_name):
        return getattr(value, attr_name)
    raise TypeError(f"Value does not expose tuple-like index {index}: {value!r}")


def _to_python_scalar(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            item = value.item()
            if isinstance(item, bool):
                return float(int(item))
            if isinstance(item, (int, float)):
                return float(item)
        except Exception:
            pass
    return None


def extract_probability(return_value: Any) -> float:
    scalar = _to_python_scalar(return_value)
    if scalar is not None:
        return float(scalar)
    probability = _get_tuple_item(return_value, 0)
    scalar = _to_python_scalar(probability)
    if scalar is None:
        raise TypeError(f"Could not extract probability from compiled SPLL return value: {return_value!r}")
    return float(scalar)


def extract_branch_count(return_value: Any) -> Optional[int]:
    try:
        metadata = _get_tuple_item(return_value, 1)
        branch_count_value = _get_tuple_item(metadata, 1)
    except TypeError:
        return None
    scalar = _to_python_scalar(branch_count_value)
    if scalar is None:
        return None
    return int(round(float(scalar)))


def make_uniform_list_read_mnist(_: str) -> List[float]:
    return [0.1] * 10


def make_skewed_list_read_mnist(_: str) -> List[float]:
    values = [0.005] * 10
    values[1] = 0.55
    values[7] = 0.405
    total = sum(values)
    return [value / total for value in values]


def make_uniform_tensor_read_mnist(_: str):
    import torch

    return torch.full((10,), 0.1)


def load_real_read_mnist(config_path: Path, model_id: Optional[str], device_name: str) -> ReadMNistFn:
    # Imported lazily so the default mock-MWE path stays as light as possible.
    from mnist_spll_common import get_model_variants, get_variant_model_output_path, load_config, resolve_device
    from mnist_spll_pipeline_core import build_read_mnist

    config = load_config(str(config_path))
    variants = get_model_variants(config)
    if model_id is None:
        if not variants:
            raise ValueError("No model variants configured.")
        model_id = str(variants[0]["id"])
    known_ids = {str(variant["id"]) for variant in variants}
    if model_id not in known_ids:
        raise ValueError(f"Unknown --model-id {model_id!r}; known variants are: {sorted(known_ids)}")
    model_path = get_variant_model_output_path(config, model_id)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}; run the train stage first.")
    device = resolve_device(device_name, False)
    return build_read_mnist(model_path, device, config_path)


def parse_sequence(raw: str) -> List[VariantName]:
    mapping = {
        "exact": "exact",
        "e": "exact",
        "cutoff": "cutoff",
        "cutoff0": "cutoff",
        "cutoff_0": "cutoff",
        "0": "cutoff",
        "0.0": "cutoff",
    }
    values: List[VariantName] = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Unsupported sequence element {item!r}; use exact or cutoff.")
        values.append(mapping[key])
    if not values:
        raise ValueError("--sequence must contain at least one block.")
    return values


def parse_image_paths(raw_values: Optional[Sequence[str]], n_terms: int) -> List[str]:
    if not raw_values:
        return [f"dummy_term_{index}.png" for index in range(int(n_terms))]
    if len(raw_values) != int(n_terms):
        raise ValueError(f"Expected exactly {n_terms} image paths, got {len(raw_values)}.")
    return [str(Path(value).resolve()) for value in raw_values]


def load_staged_experiment_image_paths(config_path: Path, n_terms: int, experiment_id: Optional[int]) -> Tuple[List[str], int, int]:
    from mnist_spll_common import load_config
    from mnist_spll_pipeline_core import build_pipeline_context, load_staged_experiments

    config = load_config(str(config_path))
    ctx = build_pipeline_context(config)
    experiments = load_staged_experiments(ctx.paths)
    chosen: Optional[Dict[str, Any]] = None
    if experiment_id is not None:
        for experiment in experiments:
            if int(experiment["experiment_id"]) == int(experiment_id):
                chosen = experiment
                break
        if chosen is None:
            raise ValueError(f"No staged experiment with experiment_id={experiment_id}.")
    else:
        for experiment in experiments:
            if int(experiment["n_terms"]) == int(n_terms):
                chosen = experiment
                break
        if chosen is None:
            raise ValueError(f"No staged experiment with n_terms={n_terms}.")
    if int(chosen["n_terms"]) != int(n_terms):
        raise ValueError(
            f"Selected staged experiment {chosen['experiment_id']} has n_terms={chosen['n_terms']}, "
            f"but --n-terms={n_terms}."
        )
    return [str(path) for path in chosen["image_paths"]], int(chosen["true_sum"]), int(chosen["experiment_id"])


def install_read_mnist(module: Any, read_mnist: ReadMNistFn) -> None:
    setattr(module, "_spll_base_readMNist", read_mnist)
    setattr(module, "readMNist", read_mnist)


def make_candidate_call(module: Any, image_paths: Sequence[str]) -> Callable[[int], Any]:
    signature = inspect.signature(module.main.forward)
    expects_acc_prob = "acc_prob" in signature.parameters

    if expects_acc_prob:
        return lambda candidate: module.main.forward(int(candidate), 1.0, *image_paths)
    return lambda candidate: module.main.forward(int(candidate), *image_paths)


def make_query(module: Any, image_paths: Sequence[str], n_terms: int, query_kind: str, true_sum: int) -> Callable[[], Dict[str, Any]]:
    candidate_call = make_candidate_call(module, image_paths)
    max_sum = 9 * int(n_terms)

    if query_kind == "true-candidate":
        def run_true_candidate() -> Dict[str, Any]:
            result = candidate_call(int(true_sum))
            return {
                "probabilities": [extract_probability(result)],
                "branch_counts": [extract_branch_count(result)],
            }

        return run_true_candidate

    if query_kind != "full-posterior":
        raise ValueError(f"Unsupported query kind: {query_kind!r}")

    def run_full_posterior() -> Dict[str, Any]:
        probabilities: List[float] = []
        branch_counts: List[Optional[int]] = []
        for candidate in range(max_sum + 1):
            result = candidate_call(candidate)
            probabilities.append(extract_probability(result))
            branch_counts.append(extract_branch_count(result))
        return {"probabilities": probabilities, "branch_counts": branch_counts}

    return run_full_posterior


def validate_probability_vector(values: Sequence[float], label: str) -> None:
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError(f"{label} produced non-finite probabilities: {values!r}")


def precheck_outputs(queries: Dict[VariantName, Callable[[], Dict[str, Any]]]) -> Dict[str, Any]:
    outputs = {name: query() for name, query in queries.items()}
    for name, payload in outputs.items():
        validate_probability_vector(payload["probabilities"], name)

    exact_values = outputs["exact"]["probabilities"]
    cutoff_values = outputs["cutoff"]["probabilities"]
    probability_abs_diffs = [abs(float(a) - float(b)) for a, b in zip(exact_values, cutoff_values)]
    exact_branches = outputs["exact"]["branch_counts"]
    cutoff_branches = outputs["cutoff"]["branch_counts"]
    branch_pairs = list(zip(exact_branches, cutoff_branches))
    branch_mismatch_count = sum(1 for exact, cutoff in branch_pairs if exact != cutoff)
    return {
        "exact_probabilities": [float(value) for value in exact_values],
        "cutoff_probabilities": [float(value) for value in cutoff_values],
        "max_probability_abs_diff": max(probability_abs_diffs) if probability_abs_diffs else 0.0,
        "branch_mismatch_count": int(branch_mismatch_count),
        "exact_branch_counts": exact_branches,
        "cutoff_branch_counts": cutoff_branches,
    }


def timed_repeat(
    *,
    block_index: int,
    variant: VariantName,
    query: Callable[[], Dict[str, Any]],
    repeats: int,
    gc_between_runs: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    checksum = 0.0
    for iteration in range(1, repeats + 1):
        if gc_between_runs:
            gc.collect()
        started_ns = time.perf_counter_ns()
        payload = query()
        elapsed_ns = time.perf_counter_ns() - started_ns
        probabilities = payload["probabilities"]
        # Keep a tiny data dependency on the result so accidental future rewrites
        # cannot silently skip the actual query call.
        checksum += float(sum(float(value) for value in probabilities))
        rows.append(
            {
                "block_index": int(block_index),
                "variant": variant,
                "iteration": int(iteration),
                "runtime_sec": elapsed_ns / 1_000_000_000.0,
                "checksum": checksum,
            }
        )
    return rows


def summarize_block(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    runtimes = [float(row["runtime_sec"]) for row in rows]
    first_count = min(10, len(runtimes))
    last_count = min(10, len(runtimes))
    return {
        "block_index": int(rows[0]["block_index"]),
        "variant": str(rows[0]["variant"]),
        "repeats": len(runtimes),
        "mean_sec": statistics.fmean(runtimes),
        "median_sec": statistics.median(runtimes),
        "stdev_sec": statistics.stdev(runtimes) if len(runtimes) >= 2 else 0.0,
        "min_sec": min(runtimes),
        "max_sec": max(runtimes),
        "first_sec": runtimes[0],
        "last_sec": runtimes[-1],
        "first_10_mean_sec": statistics.fmean(runtimes[:first_count]),
        "last_10_mean_sec": statistics.fmean(runtimes[-last_count:]),
        "warmup_ratio_first10_over_last10": (
            statistics.fmean(runtimes[:first_count]) / statistics.fmean(runtimes[-last_count:])
            if statistics.fmean(runtimes[-last_count:]) > 0
            else None
        ),
    }


def summarize_speedups(block_summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_index = {int(row["block_index"]): row for row in block_summaries}
    results: List[Dict[str, Any]] = []
    for exact in block_summaries:
        if exact["variant"] != "exact":
            continue
        for cutoff in block_summaries:
            if cutoff["variant"] != "cutoff":
                continue
            exact_mean = float(exact["mean_sec"])
            cutoff_mean = float(cutoff["mean_sec"])
            if cutoff_mean <= 0:
                continue
            results.append(
                {
                    "exact_block_index": int(exact["block_index"]),
                    "cutoff_block_index": int(cutoff["block_index"]),
                    "exact_mean_sec": exact_mean,
                    "cutoff_mean_sec": cutoff_mean,
                    "exact_over_cutoff_speedup": exact_mean / cutoff_mean,
                }
            )
    adjacent: List[Dict[str, Any]] = []
    for index in sorted(by_index):
        current = by_index[index]
        nxt = by_index.get(index + 1)
        if nxt is None or current["variant"] == nxt["variant"]:
            continue
        exact = current if current["variant"] == "exact" else nxt
        cutoff = current if current["variant"] == "cutoff" else nxt
        cutoff_mean = float(cutoff["mean_sec"])
        if cutoff_mean <= 0:
            continue
        adjacent.append(
            {
                "block_pair": [int(current["block_index"]), int(nxt["block_index"])],
                "first_variant": current["variant"],
                "exact_mean_sec": float(exact["mean_sec"]),
                "cutoff_mean_sec": cutoff_mean,
                "exact_over_cutoff_speedup": float(exact["mean_sec"]) / cutoff_mean,
            }
        )
    return [{"type": "all_pairs", "rows": results}, {"type": "adjacent_pairs", "rows": adjacent}]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def disassemble_forward(module: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(f"Disassembly for {module.__name__}.main.forward\n")
        handle.write("=" * 80 + "\n")
        try:
            dis.dis(module.main.forward, file=handle, adaptive=True, show_caches=True)
        except TypeError:
            dis.dis(module.main.forward, file=handle)


def profile_query(query: Callable[[], Dict[str, Any]], repeats: int, destination_prefix: Path) -> Dict[str, str]:
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(int(repeats)):
        query()
    profiler.disable()

    destination_prefix.parent.mkdir(parents=True, exist_ok=True)
    prof_path = destination_prefix.with_suffix(".prof")
    txt_path = destination_prefix.with_suffix(".txt")
    profiler.dump_stats(str(prof_path))

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(80)
    txt_path.write_text(stream.getvalue(), encoding="utf-8")
    return {"prof": str(prof_path), "txt": str(txt_path)}


def build_read_mnist(source: str, config_path: Optional[Path], model_id: Optional[str], device: str) -> ReadMNistFn:
    if source == "uniform-list":
        return make_uniform_list_read_mnist
    if source == "skewed-list":
        return make_skewed_list_read_mnist
    if source == "uniform-tensor":
        return make_uniform_tensor_read_mnist
    if source == "real":
        if config_path is None:
            raise ValueError("--config is required with --read-mnist-source real.")
        return load_real_read_mnist(config_path, model_id, device)
    raise ValueError(f"Unsupported readMNist source: {source!r}")


def resolve_program_paths(args: argparse.Namespace) -> Dict[VariantName, Path]:
    outputs_root = Path(args.outputs_root).resolve()
    exact_program = Path(args.exact_program).resolve() if args.exact_program else compiled_program_path(
        outputs_root, args.n_terms, args.cutoff_mode, None
    )
    cutoff_program = Path(args.cutoff_program).resolve() if args.cutoff_program else compiled_program_path(
        outputs_root, args.n_terms, args.cutoff_mode, args.cutoff
    )
    missing = [path for path in [exact_program, cutoff_program] if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Compiled SPLL program(s) missing. Run the compile stage first or pass --exact-program/--cutoff-program.\n"
            f"Missing:\n{formatted}"
        )
    return {"exact": exact_program, "cutoff": cutoff_program}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal timing harness for the exact-vs-cutoff=0.0 generated SPLL behavior. "
            "It can run with a fake readMNist list/tensor backend or the real MNIST model."
        )
    )
    default_pipeline_root = Path(__file__).resolve().parent
    parser.add_argument("--config", type=Path, default=None, help="Pipeline YAML config; required for real readMNist or staged inputs.")
    parser.add_argument("--outputs-root", type=Path, default=default_pipeline_root / "outputs")
    parser.add_argument("--exact-program", type=Path, default=None, help="Explicit exact compiled program.py path.")
    parser.add_argument("--cutoff-program", type=Path, default=None, help="Explicit cutoff=0.0 compiled program.py path.")
    parser.add_argument("--cutoff-mode", default="global")
    parser.add_argument("--cutoff", type=float, default=0.0)
    parser.add_argument("--n-terms", type=int, default=2)
    parser.add_argument("--query", choices=["full-posterior", "true-candidate"], default="full-posterior")
    parser.add_argument(
        "--read-mnist-source",
        choices=["uniform-list", "skewed-list", "uniform-tensor", "real"],
        default="uniform-list",
        help=(
            "uniform-list isolates generated Python/interpreter overhead; uniform-tensor adds PyTorch tensor dispatch; "
            "real uses the trained MNIST model."
        ),
    )
    parser.add_argument("--model-id", default=None, help="Model variant id for --read-mnist-source real.")
    parser.add_argument("--device", default="auto", help="Device for --read-mnist-source real.")
    parser.add_argument("--experiment-id", type=int, default=None, help="Use image paths and true sum from a staged experiment.")
    parser.add_argument("--image-paths", nargs="*", default=None, help="Explicit image path arguments passed to generated SPLL.")
    parser.add_argument("--true-sum", type=int, default=None, help="Candidate used by --query true-candidate.")
    parser.add_argument("--repeats", type=int, default=100, help="Number of measured calls per block.")
    parser.add_argument("--warmup", type=int, default=0, help="Untimed calls per variant before the measured sequence.")
    parser.add_argument(
        "--sequence",
        default="exact,exact,cutoff,cutoff",
        help="Comma-separated measured blocks. Example: exact,cutoff,exact,cutoff for order-effect testing.",
    )
    parser.add_argument("--gc-between-runs", action="store_true", help="Force gc.collect() before each measured call.")
    parser.add_argument("--profile-repeats", type=int, default=0, help="If >0, write cProfile reports for each variant.")
    parser.add_argument("--disassemble", action="store_true", help="Write dis.dis output for both generated forward functions.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination directory for CSV/JSON/profile artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_terms < 1:
        raise ValueError("--n-terms must be >= 1.")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")

    sequence = parse_sequence(args.sequence)
    program_paths = resolve_program_paths(args)
    modules = {name: import_compiled_module(path, name) for name, path in program_paths.items()}
    read_mnist = build_read_mnist(args.read_mnist_source, args.config, args.model_id, args.device)
    for module in modules.values():
        install_read_mnist(module, read_mnist)

    staged_experiment_id: Optional[int] = None
    if args.experiment_id is not None:
        if args.config is None:
            raise ValueError("--config is required when --experiment-id is set.")
        image_paths, staged_true_sum, staged_experiment_id = load_staged_experiment_image_paths(
            args.config, args.n_terms, args.experiment_id
        )
        true_sum = staged_true_sum if args.true_sum is None else int(args.true_sum)
    else:
        image_paths = parse_image_paths(args.image_paths, args.n_terms)
        true_sum = int(args.true_sum) if args.true_sum is not None else int(4 * args.n_terms)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(args.outputs_root).resolve() / "spll_experiments" / "diagnostics" / "cutoff_zero_mwe" / utc_now_label()
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = {
        name: make_query(module, image_paths, args.n_terms, args.query, true_sum)
        for name, module in modules.items()
    }
    precheck = precheck_outputs(queries)

    for variant, query in queries.items():
        for _ in range(int(args.warmup)):
            query()

    timing_rows: List[Dict[str, Any]] = []
    for block_index, variant in enumerate(sequence, start=1):
        block_rows = timed_repeat(
            block_index=block_index,
            variant=variant,
            query=queries[variant],
            repeats=int(args.repeats),
            gc_between_runs=bool(args.gc_between_runs),
        )
        timing_rows.extend(block_rows)

    block_summaries: List[Dict[str, Any]] = []
    for block_index in range(1, len(sequence) + 1):
        block_rows = [row for row in timing_rows if int(row["block_index"]) == block_index]
        block_summaries.append(summarize_block(block_rows))

    artifacts: Dict[str, Any] = {}
    write_csv(output_dir / "timings.csv", timing_rows)
    write_csv(output_dir / "block_summary.csv", block_summaries)
    artifacts["timings_csv"] = str(output_dir / "timings.csv")
    artifacts["block_summary_csv"] = str(output_dir / "block_summary.csv")

    if args.disassemble:
        for variant, module in modules.items():
            path = output_dir / f"dis_{variant}_forward.txt"
            disassemble_forward(module, path)
            artifacts[f"dis_{variant}"] = str(path)

    if int(args.profile_repeats) > 0:
        for variant, query in queries.items():
            artifacts[f"profile_{variant}"] = profile_query(
                query,
                int(args.profile_repeats),
                output_dir / f"profile_{variant}",
            )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "settings": {
            "n_terms": int(args.n_terms),
            "query": args.query,
            "read_mnist_source": args.read_mnist_source,
            "sequence": sequence,
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "gc_between_runs": bool(args.gc_between_runs),
            "true_sum": int(true_sum),
            "staged_experiment_id": staged_experiment_id,
            "image_paths": image_paths,
        },
        "programs": {
            variant: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for variant, path in program_paths.items()
        },
        "precheck": precheck,
        "block_summaries": block_summaries,
        "speedups": summarize_speedups(block_summaries),
        "artifacts": artifacts,
    }
    write_json(output_dir / "summary.json", summary)
    artifacts["summary_json"] = str(output_dir / "summary.json")

    print(f"Saved cutoff=0.0 MWE diagnostics to: {output_dir}")
    print(
        "Precheck: "
        f"max |p_exact - p_cutoff| = {precheck['max_probability_abs_diff']:.6g}, "
        f"branch mismatches = {precheck['branch_mismatch_count']}"
    )
    print("\nBlock summary:")
    for row in block_summaries:
        print(
            f"  block {row['block_index']:>2} {row['variant']:<6} "
            f"mean={row['mean_sec']:.6f}s median={row['median_sec']:.6f}s "
            f"first10/last10={row['warmup_ratio_first10_over_last10']:.3f}"
        )
    adjacent = next(item["rows"] for item in summary["speedups"] if item["type"] == "adjacent_pairs")
    if adjacent:
        print("\nAdjacent exact/cutoff mean ratios:")
        for row in adjacent:
            print(
                f"  blocks {row['block_pair'][0]}->{row['block_pair'][1]} "
                f"first={row['first_variant']} exact/cutoff={row['exact_over_cutoff_speedup']:.3f}"
            )


if __name__ == "__main__":
    main()
