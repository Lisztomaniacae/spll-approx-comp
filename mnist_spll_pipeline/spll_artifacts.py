from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pipeline1_config import (
    PipelinePaths,
    ThresholdSpec,
    normalize_cutoff_mode,
    threshold_label,
    threshold_spec_artifact_label,
    threshold_spec_compile_cutoff,
    threshold_spec_label,
)
from pipeline_support import TerminalProgressBar, ensure_dir


def make_spll_program(num_terms: int) -> str:
    if num_terms < 1:
        raise ValueError("num_terms must be >= 1")
    arguments = [f"x{index}" for index in range(num_terms)]
    expressions = [f"readMNist({argument})" for argument in arguments]
    expression = expressions[0]
    for next_expression in expressions[1:]:
        expression = f"({expression} ++ {next_expression})"
    return (
        "neural readMNist :: (Symbol -> Int) of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n"
        f"main {' '.join(arguments)} = {expression}\n"
    )


def ensure_programs_for_term_counts(program_root: Path, term_counts: Sequence[int]) -> None:
    ensure_dir(program_root)
    for num_terms in term_counts:
        destination = program_root / f"sum_{int(num_terms):02d}.spll"
        destination.write_text(make_spll_program(int(num_terms)), encoding="utf-8")


def compiled_program_path(
    compiled_root: Path,
    n_terms: int,
    cutoff_mode: str,
    cutoff: Optional[float],
    *,
    artifact_label: Optional[str] = None,
) -> Path:
    label = str(artifact_label) if artifact_label else threshold_label(cutoff)
    return (
        compiled_root
        / normalize_cutoff_mode(cutoff_mode)
        / f"sum_{int(n_terms):02d}"
        / label
        / "program.py"
    )


def build_compile_command(
    *,
    spll_path: Path,
    output_py_path: Path,
    cutoff: Optional[float],
    stack_arch: Optional[str],
    count_branches: bool,
) -> List[str]:
    command = ["stack"]
    if stack_arch:
        command.extend(["--arch", stack_arch])
    command.extend(["run", "--", "-i", str(spll_path)])
    if count_branches:
        command.append("-c")
    if cutoff is not None:
        command.extend(["-k", str(cutoff)])
    command.extend(["compile", "-o", str(output_py_path), "-l", "python"])
    return command


def compile_spll_program(
    *,
    repo_root: Path,
    spll_path: Path,
    output_py_path: Path,
    cutoff: Optional[float],
    force_recompile: bool,
    timeout_sec: int,
    stack_arch: Optional[str] = None,
    count_branches: bool = False,
) -> None:
    ensure_dir(output_py_path.parent)
    python_lib_source = repo_root / "pythonLib.py"
    python_lib_destination = output_py_path.parent / "pythonLib.py"

    if output_py_path.exists() and not force_recompile:
        if not python_lib_destination.exists():
            shutil.copy2(python_lib_source, python_lib_destination)
        return

    if shutil.which("stack") is None:
        raise RuntimeError("Could not find 'stack' on PATH. Install Stack before running SPLL compilation.")

    command = build_compile_command(
        spll_path=spll_path,
        output_py_path=output_py_path,
        cutoff=cutoff,
        stack_arch=stack_arch,
        count_branches=count_branches,
    )
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SPLL compilation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    shutil.copy2(python_lib_source, python_lib_destination)


def import_compiled_module(module_path: Path, module_name: str):
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_tuple_item(value: Any, index: int) -> Any:
    try:
        return value[index]
    except Exception:
        pass
    attribute = "t1" if index == 0 else "t2"
    if hasattr(value, attribute):
        return getattr(value, attribute)
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
        return scalar

    try:
        probability = _get_tuple_item(return_value, 0)
    except TypeError as exc:
        raise TypeError(
            f"Could not extract probability from compiled SPLL return value: {return_value!r}"
        ) from exc

    scalar = _to_python_scalar(probability)
    if scalar is None:
        raise TypeError(
            f"Could not extract probability from compiled SPLL return value: {return_value!r}"
        )
    return scalar


def extract_branch_count(return_value: Any) -> Optional[int]:
    """Extract the nested branch counter from ``T(probability, T(0, count))``."""

    try:
        metadata = _get_tuple_item(return_value, 1)
        branch_count_value = _get_tuple_item(metadata, 1)
    except TypeError:
        return None

    scalar = _to_python_scalar(branch_count_value)
    if scalar is None:
        raise TypeError(
            f"Could not extract branch count from compiled SPLL return value: {return_value!r}"
        )
    return int(round(scalar))


def verify_compiled_artifacts(
    paths: PipelinePaths,
    experiments: Sequence[Dict[str, Any]],
    thresholds: Sequence[ThresholdSpec],
    cutoff_modes: Sequence[str],
) -> None:
    term_counts = sorted({int(experiment["n_terms"]) for experiment in experiments})
    for cutoff_mode in cutoff_modes:
        for n_terms in term_counts:
            for threshold in thresholds:
                path = compiled_program_path(
                    paths.compiled_root,
                    n_terms,
                    cutoff_mode,
                    threshold_spec_compile_cutoff(threshold),
                    artifact_label=threshold_spec_artifact_label(threshold),
                )
                if not path.exists():
                    raise FileNotFoundError(
                        "Compiled SPLL program missing for threshold "
                        f"{threshold_spec_label(threshold)!r}: {path}. Run the 'compile' step first."
                    )


def build_compiled_module_loader(
    paths: PipelinePaths,
    cutoff_modes: Sequence[str],
    thresholds: Sequence[ThresholdSpec],
    experiments: Sequence[Dict[str, Any]],
    read_mnist,
    *,
    show_progress: bool,
):
    verify_compiled_artifacts(paths, experiments, thresholds, cutoff_modes)
    targets = sorted(
        {
            (
                normalize_cutoff_mode(mode),
                int(experiment["n_terms"]),
                threshold_spec_compile_cutoff(threshold),
                threshold_spec_artifact_label(threshold),
            )
            for experiment in experiments
            for mode in cutoff_modes
            for threshold in thresholds
        },
        key=lambda item: (
            item[0],
            item[1],
            item[3],
            item[2] is not None,
            float(item[2] or -1.0),
        ),
    )
    modules: Dict[Tuple[int, str, Optional[float], str], Any] = {}
    progress = TerminalProgressBar(
        len(targets),
        desc="Load compiled",
        unit="targets",
        enabled=show_progress and bool(targets),
    )

    def get_module(
        n_terms: int,
        cutoff_mode: str,
        cutoff: Optional[float],
        artifact_label: Optional[str] = None,
    ):
        normalized_mode = normalize_cutoff_mode(cutoff_mode)
        label = str(artifact_label) if artifact_label else threshold_label(cutoff)
        key = (int(n_terms), normalized_mode, cutoff, label)
        if key in modules:
            return modules[key]

        path = compiled_program_path(
            paths.compiled_root,
            int(n_terms),
            normalized_mode,
            cutoff,
            artifact_label=label,
        )
        digest = hashlib.sha1(str(path).encode()).hexdigest()[:10]
        module = import_compiled_module(
            path,
            f"spll_{normalized_mode}_{int(n_terms)}_{label}_{digest}",
        )
        setattr(module, "_spll_base_readMNist", read_mnist)
        setattr(module, "readMNist", read_mnist)
        modules[key] = module
        progress.update(
            postfix=f"cutoff_mode={normalized_mode}, terms={int(n_terms)}, artifact={label}"
        )
        return module

    def finish_loading() -> None:
        if targets:
            progress.finish(postfix=f"loaded {len(modules)}/{len(targets)} compiled targets used in this run")

    return get_module, finish_loading
