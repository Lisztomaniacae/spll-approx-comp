from __future__ import annotations

from typing import Any, Dict, List, Set

from pipeline1_config import (
    GLOBAL_CUTOFF_MODE,
    build_pipeline_context,
    get_configured_term_counts,
    get_thresholds,
    threshold_spec_artifact_label,
    threshold_spec_compile_cutoff,
    threshold_spec_for_json,
    threshold_spec_label,
)
from pipeline_support import (
    TerminalProgressBar,
    build_stage_metadata,
    resolve_path,
    run_configured_stage_cli,
    save_config,
    stage_message,
    write_json,
)
from spll_artifacts import (
    compile_spll_program,
    compiled_program_path,
    ensure_programs_for_term_counts,
)


def run_compile_stage(config: Dict[str, Any]) -> None:
    context = build_pipeline_context(config)
    context.paths.ensure_compile_dirs()
    repo_root = resolve_path(config, context.paths_cfg["repo_root"])
    if not repo_root.exists():
        raise FileNotFoundError(f"Configured repo_root does not exist: {repo_root}")

    thresholds = get_thresholds(config)
    term_counts = get_configured_term_counts(config)
    force_recompile = bool(context.inference_cfg.get("force_recompile", False))
    timeout_sec = int(context.inference_cfg.get("compile_timeout_sec", 600))
    stack_arch_value = context.inference_cfg.get("stack_arch")
    stack_arch = None if stack_arch_value is None else str(stack_arch_value)
    count_branches = bool(context.inference_cfg.get("count_branches", True))

    stage_message(1, 2, "Preparing SPLL source programs for configured term counts")
    ensure_programs_for_term_counts(context.paths.program_root, term_counts)
    save_config(config, context.paths.experiment_root / "compile_config_used.yaml")

    stage_message(2, 2, "Compiling SPLL programs for the global cutoff mode and every threshold")
    compile_targets: List[Dict[str, Any]] = []
    compiled_once: Set[str] = set()
    progress = TerminalProgressBar(
        len(term_counts) * len(thresholds),
        desc="Compile",
        unit="targets",
        enabled=context.show_progress and bool(term_counts) and bool(thresholds),
    )

    for n_terms in term_counts:
        spll_path = context.paths.program_root / f"sum_{n_terms:02d}.spll"
        for threshold in thresholds:
            cutoff = threshold_spec_compile_cutoff(threshold)
            label = threshold_spec_label(threshold)
            artifact_label = threshold_spec_artifact_label(threshold)
            compiled_path = compiled_program_path(
                context.paths.compiled_root,
                n_terms,
                GLOBAL_CUTOFF_MODE,
                cutoff,
                artifact_label=artifact_label,
            )
            compiled_key = str(compiled_path.resolve())
            already_compiled = compiled_key in compiled_once
            if not already_compiled:
                compile_spll_program(
                    repo_root=repo_root,
                    spll_path=spll_path,
                    output_py_path=compiled_path,
                    cutoff=cutoff,
                    force_recompile=force_recompile,
                    timeout_sec=timeout_sec,
                    stack_arch=stack_arch,
                    count_branches=count_branches,
                )
                compiled_once.add(compiled_key)

            compile_targets.append(
                {
                    "cutoff_mode": GLOBAL_CUTOFF_MODE,
                    "n_terms": int(n_terms),
                    "cutoff": cutoff,
                    "compile_cutoff": cutoff,
                    "threshold_label": label,
                    "artifact_threshold_label": artifact_label,
                    "compile_artifact_label": artifact_label,
                    "adaptive_top_k": bool(threshold.get("adaptive_top_k", False)),
                    "posterior_mass_target": threshold.get("posterior_mass_target"),
                    "cutoff_search": threshold.get("cutoff_search"),
                    "spll_path": str(spll_path),
                    "compiled_program_path": str(compiled_path),
                    "python_lib_path": str(compiled_path.parent / "pythonLib.py"),
                    "count_branches": count_branches,
                    "compiled_in_this_stage": not already_compiled,
                    "exists": compiled_path.exists(),
                }
            )
            progress.update(
                postfix=f"cutoff_mode={GLOBAL_CUTOFF_MODE}, terms={n_terms}, threshold={label}"
            )
    progress.finish(postfix="all compilation targets ready")

    write_json(
        context.paths.compile_manifest_path,
        {
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
                    "cutoff_modes": [GLOBAL_CUTOFF_MODE],
                    "thresholds": [threshold_spec_for_json(item) for item in thresholds],
                    "paths": context.paths.to_json_dict(),
                },
            ),
            "targets": compile_targets,
        },
    )
    print(f"Saved compile manifest to: {context.paths.compile_manifest_path}")
    print(f"Saved compiled artifacts under: {context.paths.compiled_root}")


def main() -> None:
    run_configured_stage_cli(
        run_compile_stage,
        description="Compile all configured SPLL programs for the MNIST addition pipeline.",
        config_help="Path to the shared YAML config.",
    )


if __name__ == "__main__":
    main()
