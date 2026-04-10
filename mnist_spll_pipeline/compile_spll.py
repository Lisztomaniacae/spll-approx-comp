from __future__ import annotations

import argparse
from typing import Any, Dict, List

from mnist_spll_common import load_config, resolve_path, set_seed, stage_message
from mnist_spll_pipeline_core import (
    build_pipeline_context,
    build_stage_metadata,
    compile_spll_program,
    ensure_programs_for_term_counts,
    get_configured_term_counts,
    get_cutoff_modes,
    get_thresholds,
    stage_config_snapshot,
    threshold_label,
    write_json,
)


def run_compile_stage(config: Dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    ctx = build_pipeline_context(config)
    repo_root = resolve_path(config, ctx.paths_cfg["repo_root"])
    if not repo_root.exists():
        raise FileNotFoundError(f"Configured repo_root does not exist: {repo_root}")

    thresholds = get_thresholds(config)
    cutoff_modes = get_cutoff_modes(config)
    term_counts = get_configured_term_counts(config)
    force_recompile = bool(ctx.inference_cfg.get("force_recompile", False))
    timeout_sec = int(ctx.inference_cfg.get("compile_timeout_sec", 600))
    stack_arch = ctx.inference_cfg.get("stack_arch")
    if stack_arch is not None:
        stack_arch = str(stack_arch)
    count_branches = bool(ctx.inference_cfg.get("count_branches", True))

    stage_message(1, 2, "Preparing SPLL source programs for configured term counts")
    ensure_programs_for_term_counts(ctx.paths.program_root, term_counts)
    stage_config_snapshot(config, ctx.paths.experiment_root / "compile_config_used.yaml")

    stage_message(2, 2, "Compiling SPLL programs for every configured cutoff mode and threshold")
    compile_targets: List[Dict[str, Any]] = []
    total_targets = len(term_counts) * len(cutoff_modes) * len(thresholds)

    from mnist_spll_common import TerminalProgressBar

    progress_bar = TerminalProgressBar(
        total_targets,
        desc="Compile",
        unit="targets",
        enabled=ctx.show_progress and total_targets > 0,
    )

    for cutoff_mode in cutoff_modes:
        for n_terms in term_counts:
            spll_path = ctx.paths.program_root / f"sum_{n_terms:02d}.spll"
            for cutoff in thresholds:
                label = threshold_label(cutoff)
                compiled_py_path = ctx.paths.compiled_root / cutoff_mode / f"sum_{n_terms:02d}" / label / "program.py"
                compile_spll_program(
                    repo_root=repo_root,
                    spll_path=spll_path,
                    output_py_path=compiled_py_path,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    force_recompile=force_recompile,
                    timeout_sec=timeout_sec,
                    stack_arch=stack_arch,
                    count_branches=count_branches,
                )
                compile_targets.append(
                    {
                        "cutoff_mode": cutoff_mode,
                        "n_terms": int(n_terms),
                        "cutoff": cutoff,
                        "threshold_label": label,
                        "spll_path": str(spll_path),
                        "compiled_program_path": str(compiled_py_path),
                        "python_lib_path": str(compiled_py_path.parent / "pythonLib.py"),
                        "count_branches": count_branches,
                        "exists": compiled_py_path.exists(),
                    }
                )
                progress_bar.update(postfix=f"cutoff_mode={cutoff_mode}, terms={n_terms}, cutoff={label}")
    progress_bar.finish(postfix="all compilation targets ready")

    manifest = {
        "metadata": build_stage_metadata(
            config,
            "compile",
            extra={
                "repo_root": str(repo_root),
                "stack_arch": stack_arch,
                "force_recompile": force_recompile,
                "compile_timeout_sec": timeout_sec,
                "count_branches": count_branches,
                "term_counts": term_counts,
                "cutoff_modes": cutoff_modes,
                "thresholds": thresholds,
                "paths": ctx.paths.to_json_dict(),
            },
        ),
        "targets": compile_targets,
    }
    write_json(ctx.paths.compile_manifest_path, manifest)
    print(f"Saved compile manifest to: {ctx.paths.compile_manifest_path}")
    print(f"Saved compiled artifacts under: {ctx.paths.compiled_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile all configured SPLL programs for the MNIST addition pipeline.")
    parser.add_argument("--config", required=True, help="Path to the shared YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_compile_stage(config)


if __name__ == "__main__":
    main()
