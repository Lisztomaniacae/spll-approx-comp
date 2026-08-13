from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

from pipeline1_config import threshold_label
from pipeline2_config import (
    TrainingPaths,
    compiled_program_path,
    get_experiments,
    get_inference_modes,
    source_program_path,
)
from pipeline_support import TerminalProgressBar, ensure_dir, resolve_path, utc_now_iso, write_json
from spll_artifacts import compile_spll_program, import_compiled_module, make_spll_program


def write_spll_sources(config: Dict[str, Any], paths: TrainingPaths) -> None:
    ensure_dir(paths.program_root)
    for experiment in get_experiments(config):
        n_terms = int(experiment["n_terms"])
        source_program_path(paths, n_terms).write_text(
            make_spll_program(n_terms),
            encoding="utf-8",
        )


def compile_training_artifacts(config: Dict[str, Any], paths: TrainingPaths) -> None:
    repo_root = resolve_path(
        config,
        config.get("paths", {}).get("repo_root", "../haskell-dppl-main"),
    )
    compile_cfg = config.get("compile", {})
    timeout_sec = int(compile_cfg.get("timeout_sec", 3600))
    stack_arch_value = compile_cfg.get("stack_arch", "x86_64")
    stack_arch = None if stack_arch_value is None else str(stack_arch_value)
    count_branches = bool(compile_cfg.get("count_branches", True))
    force_recompile = bool(compile_cfg.get("force_recompile", True))

    experiments = get_experiments(config)
    modes = get_inference_modes(config)
    write_spll_sources(config, paths)

    targets: List[Dict[str, Any]] = []
    compiled_once = set()
    progress = TerminalProgressBar(
        len(experiments) * len(modes),
        desc="Compile training SPLL",
        unit="targets",
        enabled=bool(config.get("show_progress", True)),
    )

    for experiment in experiments:
        n_terms = int(experiment["n_terms"])
        spll_path = source_program_path(paths, n_terms)
        for mode in modes:
            output_path = compiled_program_path(paths, n_terms, mode)
            compiled_key = str(output_path.resolve())
            already_compiled = compiled_key in compiled_once
            if not already_compiled:
                compile_spll_program(
                    repo_root=repo_root,
                    spll_path=spll_path,
                    output_py_path=output_path,
                    cutoff=mode.get("top_k_cutoff"),
                    force_recompile=force_recompile,
                    timeout_sec=timeout_sec,
                    stack_arch=stack_arch,
                    count_branches=count_branches,
                )
                compiled_once.add(compiled_key)

            manifest = {
                "created_at_utc": utc_now_iso(),
                "n_terms": n_terms,
                "mode_name": mode["name"],
                "artifact_name": mode.get("artifact_name", mode["name"]),
                "top_k_cutoff": mode.get("top_k_cutoff"),
                "count_branches": count_branches,
                "spll_source": str(spll_path),
                "compiled_python": str(output_path),
                "threshold_label": threshold_label(mode.get("top_k_cutoff")),
                "artifact_threshold_label": mode.get("artifact_name", mode["name"]),
                "compiled_in_this_stage": not already_compiled,
            }
            write_json(output_path.parent / "compile_manifest.json", manifest)
            targets.append(manifest)
            progress.update(postfix=f"terms={n_terms}, mode={mode['name']}")

    progress.finish(postfix="done")
    write_json(
        paths.root / "compile_manifest.json",
        {"created_at_utc": utc_now_iso(), "targets": targets},
    )


def assert_compiled_artifacts_exist(config: Dict[str, Any], paths: TrainingPaths) -> None:
    missing = [
        compiled_program_path(paths, int(experiment["n_terms"]), mode)
        for experiment in get_experiments(config)
        for mode in get_inference_modes(config)
        if not compiled_program_path(paths, int(experiment["n_terms"]), mode).exists()
    ]
    if not missing:
        return

    lines = "\n".join(f"  - {path}" for path in missing[:20])
    raise FileNotFoundError(
        "Missing compiled SPLL training artifacts. Run compile in the Rosetta/x86 environment first.\n"
        "Example:\n"
        "  arch -x86_64 zsh -f\n"
        "  source .venv-spll-x86/bin/activate\n"
        "  ./.venv-spll-x86/bin/python run_spll_training_pipeline.py "
        "--config mnist_spll_training_config.yaml compile\n"
        f"Missing artifacts:\n{lines}"
    )


def load_compiled_training_module(path: Path, n_terms: int, mode_name: str):
    digest = hashlib.sha1(str(path).encode()).hexdigest()[:10]
    return import_compiled_module(
        path,
        f"spll_training_terms_{int(n_terms):02d}_{mode_name}_{digest}",
    )
