# MNIST SPLL Pipeline Knowledge Base

This file is the living architectural memory for the Python-side MNIST + SPLL thesis pipeline.
It is intended for future assistant sessions and future human changes. Read it before changing the
pipeline.

> **Maintenance rule:** every patch and every code change that affects the MNIST SPLL pipeline MUST update this knowledge base in the same patch so future work reflects the actual current behavior.

## 1. Scope And Non-Scope

### In scope

This knowledge base covers the Python experiment pipeline under:

```text
mnist_spll_pipeline/
```

The pipeline exists to support the thesis comparison between exact and approximate inference on MNIST digit addition. The important experimental dimensions are:

- trained MNIST model variant / achieved digit accuracy;
- number of digit terms in a sum;
- exact inference versus approximate pruning threshold;
- runtime and branch-count cost;
- sum accuracy and posterior-quality metrics;
- additional true-sum-only traces, which isolate what happens to the expected candidate.

### Out of scope by default

Do **not** modify the Haskell/SPLL implementation under:

```text
haskell-dppl-main/
```

unless the user explicitly asks for it. The Python pipeline treats that folder as an external compiler/toolchain dependency. The normal interaction with it is through Stack and generated Python artifacts.

## 2. Current Architecture Snapshot

The pipeline is stage-based. The intended conceptual order is:

```text
train -> compile -> stage -> infer -> visualize
```

The current orchestration entrypoint is:

```text
mnist_spll_pipeline/run_spll_pipeline.py
```

After the architecture-deepening refactor, this entrypoint lazy-loads only the selected stage module via `load_stage_fn(...)`. This avoids importing all stage-specific dependencies when running a single stage. It does **not** yet make the compile stage dependency-free, because `compile_spll.py` still imports shared modules that themselves import PyTorch.

### Core files

| File | Role | Notes |
|---|---|---|
| `run_spll_pipeline.py` | Main stage dispatcher | Uses lazy imports. `STAGES` maps stage names to `(module_name, function_name)`. |
| `train_mnist.py` | Trains all configured MNIST model variants | Owns train/validation subset selection and checkpoint export. |
| `compile_spll.py` | Generates and compiles SPLL programs | Uses `compiled_program_path(...)` as the canonical artifact path helper. |
| `stage_experiments.py` | Samples digit-addition inputs from fixed inference split | Writes raw PNG inputs and `staged_experiments.json`. |
| `infer_experiments.py` | Inference stage orchestration | Iterates model variants and creates `InferenceRunEngine` instances. |
| `inference_engine.py` | Deepened inference-run module | Owns full-posterior query, true-candidate query, timing, and raw run schema. |
| `visualize_results.py` | Metrics, tables, and figures | Large file; contains both metric derivation and plotting. |
| `diagnose_cutoff_zero_mwe.py` | Timing diagnostic / minimal-working-example harness | Repeatedly runs exact and `cutoff=0.0` generated Python artifacts with fake or real `readMNist` backends; writes timing CSV, summaries, optional profiles, and optional disassembly. |
| `mnist_spll_pipeline_core.py` | Shared pipeline helpers | Paths, SPLL program generation, compilation wrapper, module loading, inference helpers, JSON helpers. |
| `mnist_spll_common.py` | Shared config/model/data utilities | Config loading, path resolution, CNN model, MNIST loading, model variant config. |
| `run_spll_sum_experiments.py` | Legacy wrapper | Kept for compatibility, but should not be the preferred entrypoint. See known caveat below. |
| `mnist_spll_config.yaml` | Joint config | Drives all stages. Some old-looking keys may be inert; see config caveats. |

## 3. High-Level Data Flow

```text
mnist_spll_config.yaml
        |
        v
train_mnist.py
        |-- outputs/models/mnist_splits.pt
        |-- outputs/models/model_<variant>.pt
        |-- outputs/models/model_selection_manifest.json
        |
        v
compile_spll.py
        |-- outputs/spll_experiments/generated/spll_programs/sum_XX.spll
        |-- outputs/spll_experiments/generated/compiled_python/global/sum_XX/<artifact_threshold_label>/program.py
        |-- outputs/spll_experiments/compile_manifest.json
        |
        v
stage_experiments.py
        |-- outputs/spll_experiments/inputs/experiment_XXXX/*.png
        |-- outputs/spll_experiments/staged_experiments.json
        |
        v
infer_experiments.py + inference_engine.py
        |-- outputs/spll_experiments/inference_manifest.json
        |-- outputs/spll_experiments/inference_runs.json
        |
        v
visualize_results.py
        |-- outputs/spll_experiments/visualization/tables/*.csv
        |-- outputs/spll_experiments/visualization/tables/*.json
        |-- outputs/spll_experiments/visualization/figures/**/*.png
```

`outputs/spll_experiments/inference_runs.json` is the most important raw empirical artifact. Visualization should derive metrics from it instead of rerunning inference.

## 4. Environment And Platform Workflow

The project is normally used on Apple Silicon with two environments:

| Stage | Preferred environment | Reason |
|---|---|---|
| `train` | native arm64 Python venv | Uses PyTorch/MPS for model training. |
| `compile` | Rosetta/x86_64 shell and venv | Stack/GHC/SPLL compilation has historically been more reliable here. |
| `stage` | native arm64 Python venv | Uses torchvision/Pillow/MNIST data handling. |
| `infer` | native arm64 Python venv | Uses PyTorch/MPS for MNIST neural callback during compiled SPLL execution. |
| `visualize` | native arm64 Python venv | Uses NumPy/matplotlib and saved JSON/CSV data. |

### Important environment caveats

1. Use explicit interpreter paths, not a shell alias such as `python -> /opt/homebrew/bin/python@3.11`.
2. Lazy stage loading means `run_spll_pipeline.py compile` no longer imports every stage file. However, the compile stage still imports shared modules that currently import `torch`, so the x86 compile environment may still need PyTorch installed until shared imports are split further.
3. `stack` must be available on `PATH` for the compile stage.
4. The compile stage uses `paths.repo_root` from YAML to find the local SPLL/NeST checkout and `pythonLib.py`.
5. Running `all` in one command is less robust on Apple Silicon because stages prefer different CPU architectures/environments. Prefer explicit stage-by-stage commands.

### Canonical commands

From `mnist_spll_pipeline/`:

```bash
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml train
```

```bash
arch -x86_64 zsh -f
source .venv-spll-x86/bin/activate
./.venv-spll-x86/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml compile
```

```bash
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml stage
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml infer
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml visualize
```

## 5. Config Semantics And Caveats

The config is loaded by `mnist_spll_common.load_config(...)`. Loading injects two internal keys:

- `_config_path`: absolute path to the YAML file;
- `_config_dir`: parent directory of the YAML file.

Path resolution should go through `resolve_path(config, raw_path)`, so relative paths stay relative to the YAML file rather than the current shell directory.

### Inference config

Important keys under `inference:`:

| Key | Meaning |
|---|---|
| `num_experiments` | Number of staged MNIST sum experiments sampled from the fixed inference split. |
| `terms_per_sum_min`, `terms_per_sum_max` | Inclusive term-count range. A separate SPLL program is generated for each term count. |
| `sample_without_replacement_within_experiment` | If true, one staged sum does not reuse the same inference example twice. |
| `top_predictions_to_store` | Number of top posterior candidates serialized in detailed visualization rows. |
| `approximation_thresholds` | List of threshold specs. `null` means exact/no pruning, numeric values mean fixed approximate pruning thresholds, and mapping entries with `top_k_cutoff: auto` enable adaptive posterior-mass cutoff tuning. |
| `adaptive_top_k` | Optional defaults for Pipeline I adaptive threshold entries: `posterior_mass_target`, `probe_experiments`, `max_iterations`, `tolerance`, `min_cutoff`, and `max_cutoff`. |
| `count_branches` | If true, SPLL compilation uses `-c` and compiled probability calls return branch metadata. |
| `force_recompile` | If false and a compiled `program.py` already exists, compile can skip regeneration. |
| `compile_timeout_sec` | Timeout for the Stack compilation subprocess. |
| `stack_arch` | Optional architecture argument for Stack, e.g. `x86_64`. |
| `show_progress`, `show_inner_progress` | Control outer and inner terminal progress bars. |

### Approximation threshold semantics

The SPLL compiler flag `-k/--topKCutoff` is a probability cutoff in `[0, 1]`. It is **not** a literal top-k count.

Current interpretation:

- `null` in YAML -> exact baseline, no pruning flag passed to SPLL;
- `0.0` -> approximate code path with zero cutoff; useful for measuring overhead of the approximation machinery versus exact;
- positive numbers such as `0.01`, `0.05`, `0.1`, `0.25` -> fixed approximate pruning runs;
- mapping entries with `top_k_cutoff: auto`, `adaptive_top_k: true`, and `posterior_mass_target` -> adaptive approximate runs. They compile through SPLL's approximate code path with seed cutoff `0.0`, into the dedicated artifact label `cutoff_topk`, and tune the generated module-level `TOP_K_CUTOFF` at runtime.

Pipeline I adaptive threshold example:

```yaml
inference:
  adaptive_top_k:
    posterior_mass_target: 0.8
    probe_experiments: 20
    max_iterations: 14
    tolerance: 0.02
  approximation_thresholds:
    - null
    - 0.01
    - name: approx_mass_0p8
      top_k_cutoff: auto
      adaptive_top_k: true
      posterior_mass_target: 0.8
```

Pipeline I tuning is deterministic bounded monotone bisection over cutoff values. It runs once per `(model_id, cutoff_mode, n_terms, adaptive threshold label)` using the first configured staged probe experiments for that term count. The search objective is mean unnormalised posterior mass that survives pruning across full candidate-sum enumeration. Timed posterior inference remains separate from cutoff-search runtime, and raw runs log `runtime_top_k_cutoff`, `mean_surviving_posterior_mass`, `adaptive_cutoff_search_runtime_sec`, the complete `adaptive_cutoff_state`, and visualizable bisection rows in `visualization/tables/adaptive_topk_search_trace.*`. The Pipeline I adaptive top-k search figures share one figure-level legend across all term-count panels; do not reintroduce per-panel duplicate legends unless the plotted series differ by panel. Adaptive thresholds compile into their own `cutoff_topk` artifact directory for each term count, while numeric `0.0` remains `cutoff_0p0`; `InferenceRunEngine.run_one` still resets `TOP_K_CUTOFF` before every fixed-cutoff run so runtime state cannot leak across measurements.

### Cutoff mode caveat

The Python pipeline currently uses only the accumulated global path-mass cutoff. `get_cutoff_modes(config)` always returns:

```python
["global"]
```

The YAML may still contain a `cutoff_modes:` key for historical reasons, but it is no longer a real user-facing knob. Do not reintroduce local/global branching unless the experiment design explicitly needs it and all artifact paths, visualization grouping, and documentation are updated.

### Training config and biased variants

Training uses configured `model_variants`. Each variant is trained and the exported checkpoint is chosen by the epoch whose validation accuracy is closest to `target_accuracy`.

Current biased variants use:

- an extreme broken-prior training distribution;
- a uniform validation distribution;
- the same global inference holdout as other variants.

This design intentionally keeps inference comparable across biased and unbiased models while making the trained model priors distorted.

### Naming caveat: test versus validation

The code often uses `test` names for what is functionally the model-selection validation split. For example:

- config key `test_ratio` controls the validation/model-selection split;
- checkpoint field `selected_test_accuracy` means selected validation accuracy;
- split manifest key `test_indices` means validation indices.

When writing thesis text, prefer “validation/model-selection accuracy” unless the artifact field name is being discussed directly.

## 6. Stage Details

### 6.1 Training stage

Entrypoint:

```text
train_mnist.run_training(config)
```

Main responsibilities:

1. seed Python/NumPy/PyTorch;
2. resolve device, usually MPS on Apple Silicon;
3. load full MNIST train+test as one concatenated dataset;
4. split it into train, validation, and inference subsets using configured ratios and seed;
5. build label-index pools for train and validation subsets;
6. train every configured model variant;
7. save a split manifest, per-variant checkpoints, per-epoch metrics CSVs, and a model-selection manifest.

Important training functions:

| Function | Role |
|---|---|
| `compute_split_lengths(...)` | Validates ratios and computes train/validation/inference sizes. |
| `get_model_variants(...)` | Normalizes/inherits model variant config. |
| `select_variant_subset(...)` | Applies per-variant label distribution and max-example constraints. |
| `train_variant(...)` | Trains one model variant and exports selected checkpoint. |
| `choose_epoch_nearest_target(...)` | Chooses checkpoint by closest validation accuracy to target. |

Training caveats:

- Target accuracy is a selection target, not a guarantee.
- Low-accuracy variants are produced mostly through smaller models, limited examples, and early/nearest checkpoint selection.
- `require_mps: true` will fail if MPS is unavailable.
- Validation can be distribution-controlled separately from training, which is important for biased variants.

### 6.2 Compile stage

Entrypoint:

```text
compile_spll.run_compile_stage(config)
```

Main responsibilities:

1. derive normalized threshold specs, term counts, and fixed cutoff mode;
2. write one SPLL source program per term count;
3. compile each unique `(cutoff_mode, n_terms, compile_cutoff, artifact_label)` artifact; adaptive posterior-mass thresholds use artifact label `cutoff_topk`;
4. copy `pythonLib.py` beside each generated `program.py`;
5. write `compile_manifest.json`.

Adaptive Pipeline I thresholds have a distinct user-facing `threshold_label` such as `approx_mass_0p8`, but their `compile_cutoff` is `0.0` and their compiled artifact label is `cutoff_topk`. This keeps the adaptive generated artifact separate from fixed numeric `cutoff_0p0`; runtime code is still responsible for setting/resetting `TOP_K_CUTOFF` for the actual measurement.

Canonical compiled artifact path:

```text
outputs/spll_experiments/generated/compiled_python/global/sum_XX/<artifact_threshold_label>/program.py
```

The canonical path helper is:

```python
compiled_program_path(compiled_root, n_terms, cutoff_mode, cutoff)
```

Do not duplicate this path formula in callers.

SPLL program generation currently creates explicit left-associated addition:

```spll
main x0 x1 x2 = ((readMNist(x0) ++ readMNist(x1)) ++ readMNist(x2))
```

Keep this explicit unless SPLL parser/semantics are rechecked. Parser or operator-associativity assumptions should not be guessed.

Compilation command shape:

```bash
stack [--arch x86_64] run -- -i <sum_XX.spll> [-c] [-k <cutoff>] compile -o <program.py> -l python
```

Compile caveats:

- `null` cutoff does not pass `-k`.
- `count_branches: true` passes `-c`.
- If `force_recompile` is false and `program.py` exists, compilation is skipped; `pythonLib.py` is still copied if missing.
- The Python pipeline does not inspect or modify Haskell internals during this stage.

### 6.3 Stage-experiments stage

Entrypoint:

```text
stage_experiments.run_stage_experiments(config)
```

Main responsibilities:

1. load the fixed inference split from `mnist_splits.pt`;
2. load raw MNIST images without tensor transforms;
3. sample `num_experiments` term counts uniformly from the configured inclusive term range;
4. sample MNIST examples from the inference split;
5. save each selected image as a PNG under `inputs/experiment_XXXX/`;
6. write `staged_experiments.json`.

Each staged experiment contains:

```json
{
  "experiment_id": 1,
  "n_terms": 3,
  "global_indices": [123, 456, 789],
  "image_paths": [".../term_00_...png", "..."],
  "labels": [5, 7, 1],
  "true_sum": 13
}
```

Staging caveats:

- The staged experiments are sampled from the fixed inference split, not from train or validation.
- `true_sum` is the sum of the ground-truth MNIST labels, not the model prediction.
- If `sample_without_replacement_within_experiment` is true, one experiment will not use the same inference sample twice, but different experiments may reuse the same sample.

### 6.4 Inference stage

Entrypoint:

```text
infer_experiments.run_inference_stage(config)
```

Deepened boundary:

```text
inference_engine.InferenceRunEngine
```

Main responsibilities split:

| Layer | Responsibility |
|---|---|
| `infer_experiments.py` | Load model variants, checkpoints, staged experiments, device, compiled-module loader, and write final manifests. |
| `inference_engine.py` | Execute one or many inference runs and produce stable raw run records. |
| `mnist_spll_pipeline_core.py` | Load compiled modules, patch `readMNist`, evaluate candidate sums, extract probabilities/branch counts. |

`build_read_mnist(...)` loads the selected MNIST checkpoint and returns a deliberately **uncached** callback that maps image paths to class probabilities. The compiled SPLL module is imported dynamically, then its `readMNist` symbol is replaced with this callback and the base callback is preserved as `_spll_base_readMNist`.

Inference timing is controlled by `inference.read_mnist_cache_policy` in `mnist_spll_config.yaml`. The canonical values are:

| Value | Meaning |
|---|---|
| `uncached` | Default thesis timing policy. Do not cache generated-SPLL neural calls; each `readMNist` invocation calls the base MNIST model, while still recording call/miss/unique-image stats. This measures the literal generated-SPLL execution without a memoized neural oracle. |
| `run_scoped_no_cross_run_cache` | Sensitivity/debug policy. Install a fresh query-scoped `readMNist` cache around each measured generated-SPLL call. This avoids repeated neural forward passes for the same image inside one posterior query, but prevents one cutoff measurement from warming the cache for another cutoff measurement. |
| `precomputed_per_measurement` | Inference-isolation policy. For each measured full-posterior or true-candidate query, compute neural probabilities for that run's image paths into a fresh lookup immediately before timing, install a lookup-only `readMNist`, time only the generated-SPLL query, then discard the lookup. Precompute time is logged separately and is never shared between exact and cutoff runs. |

Do not reintroduce a process-global `@lru_cache` around `read_mnist`; it can make later thresholds, especially `0.0`, look artificially faster than exact. Use `run_scoped_no_cross_run_cache` only when the research question explicitly wants to factor out repeated identical CNN calls inside one generated-SPLL query. Use `precomputed_per_measurement` when the research question wants to isolate generated probabilistic-inference/runtime behavior after neural predictions are already available.

`inference.read_mnist_warmup_calls` controls optional untimed base `readMNist` calls before each model variant starts measured inference. The benchmark default is `0`. Keep it at `0` for thesis-facing runtime results so the measured run includes the same cold-start behavior a normal uncached inference call would see. Use a positive value only for diagnostics that intentionally factor out one-off PIL/PyTorch/device-dispatch startup effects.

For each model variant, staged experiment, cutoff mode, and threshold, inference does two conceptually separate measurements:

1. **Full posterior query:** evaluate every candidate sum from `0` through `9 * n_terms` inside one fresh readMNist scope using the selected cache policy.
2. **True-candidate query:** additionally evaluate only the expected true sum, e.g. for labels `5` and `7`, evaluate candidate `12` alone inside a separate fresh readMNist scope using the selected cache policy.

The full posterior runtime is stored as `runtime_sec`. The true-candidate-only runtime is stored separately as `true_candidate_runtime_sec`. Do not add them together unless you explicitly want total measurement overhead of the instrumentation.

Thresholds are measured in a rotated order by experiment index so `exact` is not always first. Raw run records include both the measurement order and the threshold order position used for that run.

Raw run schema highlights:

| Field | Meaning |
|---|---|
| `model_id` | Configured model variant ID. |
| `target_accuracy` | Target accuracy from config. |
| `selected_epoch` | Epoch selected for exported checkpoint. |
| `selected_test_accuracy` | Selected validation accuracy, despite the field name. |
| `experiment_id` | Staged experiment ID. |
| `cutoff_mode` | Currently always `global`. |
| `n_terms` | Number of MNIST digits in the sum. |
| `cutoff` | `null`, `0.0`, or positive cutoff. |
| `threshold_label` | User-facing run label such as `exact`, `cutoff_0p01`, or `approx_mass_0p8`. |
| `artifact_threshold_label` | Compiled artifact directory label; adaptive posterior-mass runs use `cutoff_topk`. |
| `candidate_sums` | All queried candidate sums, normally `[0, ..., 9 * n_terms]`. |
| `posterior_raw` | Raw, unnormalized probability/mass for each candidate. |
| `branch_counts_raw` | Branch count per candidate, or `null` values if unavailable. |
| `runtime_sec` | Full posterior query runtime only. |
| `true_candidate_sum` | Ground-truth sum queried separately. |
| `true_candidate_probability_raw` | Raw probability for the true-sum-only query. |
| `true_candidate_branch_count` | Branch count for the true-sum-only query. |
| `true_candidate_runtime_sec` | Runtime for the true-sum-only query. |
| `labels`, `global_indices`, `image_paths` | Provenance for the staged input. |
| `compiled_program_path` | Exact compiled program used. |
| `compiled_program_sha256` | SHA-256 digest of the compiled generated Python file used by the run. |
| `measurement_order_index` | Actual chronological position of this timed threshold measurement in the inference stage. |
| `threshold_order_position` | Position of the threshold within the rotated per-experiment threshold order. |
| `read_mnist_cache_policy` | Cache policy used for generated SPLL neural calls: `uncached`, `run_scoped_no_cross_run_cache`, or `precomputed_per_measurement`. |
| `read_mnist_warmup_calls`, `read_mnist_warmup_runtime_sec` | Optional untimed warmup metadata for the base `readMNist` callback. Thesis-facing benchmark runs should normally record `0` calls and `0.0` seconds. |
| `posterior_read_mnist_stats` | Calls, hits, misses, and unique image count for the full-posterior readMNist scope. |
| `true_candidate_read_mnist_stats` | Calls, hits, misses, and unique image count for the true-candidate-only readMNist scope. |
| `read_mnist_precompute_runtime_sec` | Full-posterior neural precompute time when `precomputed_per_measurement` is used; otherwise `0.0`. This is deliberately excluded from `runtime_sec`. |
| `true_candidate_read_mnist_precompute_runtime_sec` | True-candidate neural precompute time when `precomputed_per_measurement` is used; otherwise `0.0`. This is deliberately excluded from `true_candidate_runtime_sec`. |

Branch-count extraction caveat:

Compiled SPLL probability calls with branch counting enabled are expected to return a nested tuple-like value shaped approximately like:

```text
T(probability, T(0.0, branch_count))
```

`extract_branch_count(...)` reads the nested second component. If the structure is missing, it returns `None`. If the structure exists but cannot be converted to a scalar, it raises.

### 6.5 Cutoff-zero timing diagnostic / MWE harness

`diagnose_cutoff_zero_mwe.py` exists specifically to investigate suspicious speedups of the approximate `-k 0.0` generated Python path. It is not part of the main pipeline stages and does not write `inference_runs.json`; it writes an isolated diagnostic bundle under:

```text
outputs/spll_experiments/diagnostics/cutoff_zero_mwe/<timestamp>/
```

The harness loads the already compiled exact artifact and the already compiled `cutoff_0p0` artifact for a chosen arity. It then installs one selected `readMNist` backend into both modules:

| Backend | Purpose |
|---|---|
| `uniform-list` | Replaces MNIST with `[0.1] * 10`; isolates generated Python and CPython interpreter behavior. |
| `skewed-list` | Uses a deterministic nonuniform Python-list distribution; still avoids PyTorch. |
| `uniform-tensor` | Returns a uniform torch tensor; adds PyTorch tensor dispatch while still avoiding CNN execution and image IO. |
| `real` | Uses `build_read_mnist(...)` with a configured checkpoint and staged image paths. |

Useful commands:

```bash
./.venv-train-arm64/bin/python diagnose_cutoff_zero_mwe.py \
  --config mnist_spll_config.yaml \
  --n-terms 2 \
  --read-mnist-source uniform-list \
  --query full-posterior \
  --repeats 100 \
  --sequence exact,exact,cutoff,cutoff \
  --disassemble \
  --profile-repeats 1000
```

```bash
./.venv-train-arm64/bin/python diagnose_cutoff_zero_mwe.py \
  --config mnist_spll_config.yaml \
  --n-terms 2 \
  --experiment-id 1 \
  --model-id acc90 \
  --read-mnist-source real \
  --query full-posterior \
  --repeats 100 \
  --sequence exact,cutoff,exact,cutoff
```

Diagnostic artifacts:

| File | Meaning |
|---|---|
| `timings.csv` | One measured row per generated-SPLL call, including total runtime plus timed `readMNist` backend time, residual generated-SPLL/inference time, call counts, unique image counts, and time shares. |
| `block_summary.csv` | Mean/median/stdev/min/max plus first-10 versus last-10 timing ratios per sequence block; also aggregates the `readMNist` versus residual inference split. |
| `adjacent_time_savings.csv`, `all_pair_time_savings.csv` | Exact/cutoff pair decompositions: total speedup, inference-only speedup, `readMNist` runtime ratio, saved seconds by component, and the maximum exact-time share that could be saved by inference-only improvements. |
| `summary.json` | Settings, environment metadata, program SHA-256 values, precheck probability/branch differences, block summaries, speedup ratios, and adjacent-pair savings decomposition. |
| `profile_exact.txt`, `profile_cutoff.txt` | Optional `cProfile` summaries when `--profile-repeats` is positive. |
| `dis_exact_forward.txt`, `dis_cutoff_forward.txt` | Optional bytecode disassembly when `--disassemble` is set. |

Interpretation discipline:

- If a speedup persists with `uniform-list`, the effect is in generated Python / CPython behavior, not the CNN or PyTorch model execution.
- If it appears only with `uniform-tensor`, suspect torch tensor dispatch/autograd-related overhead rather than image IO or the trained model.
- If it appears only with `real`, inspect `readMNist`, image loading, device warmup, or cache policy.
- Use `mean_read_mnist_runtime_sec` versus `mean_inference_runtime_sec` to estimate how much measured wall time is neural backend work and how much is potentially saveable generated-SPLL/inference work. For exact/cutoff block pairs, `max_inference_only_savable_share_of_exact_total` is the rough upper bound on speedup available from making symbolic inference free while leaving MNIST calls unchanged.
- The default sequence `exact,exact,cutoff,cutoff` tests first-block versus second-block warmup inside each artifact. Use `exact,cutoff,exact,cutoff` to test cross-artifact run-order effects.

### 6.6 Visualization stage

Entrypoint:

```text
visualize_results.run_visualization_stage(config)
```

Main responsibilities:

1. read `staged_experiments.json` and `inference_runs.json`;
2. derive detailed rows from raw runs;
3. summarize by cutoff mode, model, term count, and threshold;
4. add exact-baseline deltas;
5. write CSV/JSON tables;
6. write main-text and appendix figures.

Important metric derivation functions:

| Function | Role |
|---|---|
| `prepare_detailed_rows(...)` | Converts raw run records into per-run metrics. |
| `summarize_groups(...)` | Aggregates detailed rows into summary metrics. |
| `add_exact_baseline_columns(...)` | Adds speedup, runtime ratio, accuracy delta, branch-count delta, and true-candidate deltas versus exact. |
| `heatmap_specs()` | Defines appendix heatmaps. |
| `plot_*` functions | Produce specific figure files. |

Core detailed metrics:

| Metric | Meaning |
|---|---|
| `predicted_sum` | Candidate with maximum normalized posterior probability. |
| `correct` | Whether `predicted_sum == true_sum`. |
| `confidence` | Normalized posterior probability of `predicted_sum`. |
| `posterior_mass` | Sum of raw posterior values before normalization. |
| `posterior_entropy` | Entropy of normalized posterior. |
| `output_pool` | Count of candidates with raw mass greater than `EPS`. |
| `output_pool_fraction` | `output_pool / candidate_count`. |
| `total_branch_count` | Sum of available per-candidate branch counts. |
| `zero_mass` | Whether posterior mass collapsed to zero. |
| `true_candidate_survived` | Whether raw true-candidate probability is greater than `EPS`. |
| `true_candidate_branch_fraction_of_total` | True-candidate branch count divided by total branch count, when defined. |
| `true_candidate_runtime_fraction_of_full` | True-candidate runtime divided by full posterior runtime, when defined. |

Collapse-rate caveat:

In the current visualization code, “collapse rate” means the average of `zero_mass`, i.e. the fraction of runs whose full raw posterior mass is zero or less. It is **not** the complement of accuracy, survival rate, or output-pool fraction.

Exact-baseline caveat:

Exact-baseline deltas are computed within the same `(cutoff_mode, model_id, n_terms)` group. If the exact row is missing for a group, deltas become `NaN`.

Visualization layout notes:

- The main runtime–accuracy tradeoff figure is split by term count and is plotted relative to exact inference, not on raw accuracy/runtime axes. Each panel uses `runtime / exact_runtime` on the x-axis and accuracy delta in percentage points versus exact on the y-axis, with exact shown once as the `(1, 0)` reference. This keeps the plot focused on the approximation tradeoff instead of mixing it with the baseline quality differences between trained models.
- Runtime-vs-cutoff figures keep cutoff on the numeric x-axis. Fixed cutoff runs use their configured cutoff value, while adaptive posterior-mass runs use the selected runtime `TOP_K_CUTOFF` and a visibly larger star marker.
- Full-posterior and true-sum-only runtime-vs-cutoff figures draw dashed horizontal exact-runtime reference lines for each model in the same color as that model's approximate series. These are visual references only; the exact rows remain outside the cutoff x-axis.
- The target-vs-achieved model-accuracy figure is vertical by metric: models sit on the x-axis and accuracy sits on the y-axis, with paired target/achieved markers connected per model.
- Dense line/scatter figures should prefer figure-level horizontal legends in reserved space above or below the panels instead of repeated per-panel legends. Keep legend anchors inside reserved figure margins; for bottom legends, use upper-anchored legend boxes inside the reserved bottom band so stacked legend rows cannot grow into each other or into the subplot grid when `bbox_inches="tight"` is used.

## 7. Current Known Caveats / Sharp Edges

Keep these visible. They are useful future patch targets.

1. **Legacy wrapper likely needs update after lazy stage loading.** `run_spll_sum_experiments.py` imports `STAGES` from `run_spll_pipeline.py`, but `STAGES` now contains `(module_name, function_name)` tuples rather than callable functions. Prefer `run_spll_pipeline.py`; if the legacy wrapper must be kept working, update it to use `load_stage_fn(...)` and update this knowledge base in that same patch.
2. **Shared imports are still too heavy.** `compile_spll.py` lazy-loads as a stage, but imports `mnist_spll_pipeline_core.py` and `mnist_spll_common.py`, which import PyTorch. The compile environment may still need PyTorch until compile-only helpers are separated from neural/data helpers.
3. **Visualization is still shallow/large.** `visualize_results.py` combines metric derivation, aggregation, plotting style, plot layout, and output writing. Future changes should deepen metrics before adding more plot-specific complexity.
4. **Raw config dict remains the main cross-stage interface.** A future `ExperimentPlan` object would reduce string-key coupling and make tests cleaner.
5. **`selected_test_accuracy` naming is misleading.** It means selected validation/model-selection accuracy.
6. **True-candidate tracing is extra work.** It is intentionally separate data, not part of the original posterior run. Keep `runtime_sec` and `true_candidate_runtime_sec` separate.
7. **`0.0` cutoff is not exact.** It is useful as an approximate-code-path overhead baseline, but exact is represented only by `null`. After the timing-cache patch, `0.0` should no longer benefit from an exact-warmed process-global readMNist cache.
8. **Do not restore process-global readMNist caching.** Repeated neural calls may be cached only inside a single measured query scope through `run_scoped_no_cross_run_cache`, deliberately disabled through the default `inference.read_mnist_cache_policy: uncached`, or precomputed into a fresh per-measurement lookup through `precomputed_per_measurement`. Cross-run caching contaminates exact-vs-cutoff comparisons and can manufacture speedups.
9. **Runtime plots must state the readMNist policy.** Use `uncached` for literal end-to-end generated-SPLL execution. Use `precomputed_per_measurement` for inference-isolated timing. Do not mix the two regimes in one speedup claim without labeling them.
10. **Do not hide cold-start effects in thesis-facing benchmarks.** Keep `read_mnist_warmup_calls: 0` for normal uncached runtime results. Use positive warmup only for explicit diagnostics that ask whether a one-off startup effect is present.
11. **Use the cutoff-zero MWE harness before explaining suspicious `0.0` speedups.** Prefer `uniform-list` first, then `uniform-tensor`, then `real`, so generated Python/interpreter effects are separated from PyTorch and image/model effects.
12. **Generated SPLL syntax should be treated conservatively.** Do not change parenthesization or operator shape without verifying SPLL parser/compiler behavior.
13. **Manifests are part of reproducibility.** If a stage changes schema, update the knowledge base and preferably include backward-compatible visualization fallback where cheap.
14. **Branch-count timing caveat.** Branch counting is currently treated as symmetric enough for the exact-vs-approximate comparison, but thesis-facing runtime text should state when `count_branches: true` was enabled. If clean runtime becomes central, run a non-counting timing sensitivity check rather than assuming instrumentation overhead is exactly equal across modes.
15. **Pipeline I uses a custom MNIST-pool split.** This is intentional: official MNIST train and test partitions are loaded into one pool and then split into train, validation/model-selection, and inference subsets by the configured ratios and seed. Do not describe Pipeline I as an official MNIST-test benchmark unless the split logic is changed.
16. **Large-N random sampling is the current mitigation for staged-case luck.** The final run is expected to use many more staged experiments rather than stratified balancing. If residual fairness concerns remain, add slice reporting by arity, true sum, model variant, collapse rate, and exact confidence.
17. **Operand-order sensitivity is known but not currently patched.** Generated addition expressions are left-associated; approximation may be branch/order sensitive. Large-N random sampling is the current practical mitigation. A future permutation-sensitivity experiment would be the cleanest direct check.

## 8. Patch Discipline For Future Work

Before making a patch:

1. Read this file.
2. Identify which stage or boundary owns the requested behavior.
3. Avoid touching `haskell-dppl-main/` unless explicitly requested.
4. Prefer changing one deep boundary over scattering small helper edits across stages.
5. Preserve artifact schemas unless the user explicitly accepts migration cost.
6. Decide whether old outputs need deletion/regeneration after the patch.

Every patch should include:

- code changes;
- this file updated if behavior, workflow, schemas, commands, caveats, or architecture changed;
- verification notes for what was checked;
- a clear `git apply` command in the final response.

Recommended verification commands from repo root:

```bash
git apply --check <patch-file.patch>
```

```bash
python - <<'PY'
import ast
from pathlib import Path
for path in Path('mnist_spll_pipeline').glob('*.py'):
    ast.parse(path.read_text())
print('all mnist_spll_pipeline/*.py files parse')
PY
```

When local dependencies are available, prefer stage-level smoke checks over helper-only checks:

```bash
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml stage
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml visualize
```

Only run heavy training/inference if the user asks or if the change genuinely requires it.

## 9. Architectural Design Guidelines

Use John Ousterhout-style deep modules: small interface, substantial hidden implementation.

### Good direction

- A stage script should orchestrate one stage, not own low-level schema details.
- A deep module should own a stable boundary, such as “execute inference run and produce raw record.”
- Callers should not duplicate path formulas, threshold labels, or raw JSON field construction.
- Tests should assert behavior at module boundaries, not tiny internal helper details.

### Avoid

- Adding another helper function just to make one line testable while leaving the real call sequence untested.
- Passing raw config dicts deeper than necessary.
- Duplicating artifact paths in multiple files.
- Letting visualization-specific choices change metric semantics silently.
- Renaming artifact fields casually; saved JSON/CSV files are analysis inputs.

### Current best refactor targets

1. **Experiment plan module.** Convert raw YAML config into a typed `ExperimentPlan` containing paths, thresholds, term counts, model variants, and compile targets.
2. **Visualization metrics module.** Split metric derivation from plotting so tables can be boundary-tested without matplotlib.
3. **Compile artifact manager.** Move SPLL source generation, artifact pathing, Stack invocation, and manifest rows behind a small compile API.
4. **Stage experiment repository.** Represent staged experiments as typed records instead of ad-hoc dicts.
5. **Training variant planner.** Deepen biased/unbiased subset selection and checkpoint-selection semantics into a testable plan.

## 10. Testing Strategy Guidelines

Follow the principle: replace, do not layer.

When a deep boundary test exists, delete or avoid redundant tests that only mirror internal helper steps.

Suggested boundary tests:

| Boundary | Test idea |
|---|---|
| `InferenceRunEngine` | Use a fake compiled module and fake module loader to assert full-posterior fields, true-candidate fields, branch-count handling, and runtime fields exist. |
| Config/experiment plan | Given a small YAML fixture, assert thresholds, labels, paths, term counts, model variants, and compile targets. |
| Staging | With a tiny fake dataset and temp directory, assert deterministic staged JSON shape and correct `true_sum`. |
| Compile artifact manager | With fake command runner/temp repo, assert generated source, output paths, skip behavior, `pythonLib.py` copy, and manifest rows. |
| Visualization metrics | Given tiny raw runs, assert detailed and summary metrics, exact deltas, collapse rate, and true-candidate metrics without creating plots. |

Prefer local stand-ins:

- fake compiled module with `main.forward(...)`;
- temp filesystem roots;
- tiny fake datasets;
- small JSON fixtures.

Avoid real MNIST downloads, real Stack compilation, and full model training in unit tests.

## 11. Formatting Guidelines For This Knowledge Base

Use predictable Markdown so future assistants can patch it safely.

### Structure rules

- Keep top-level sections numbered with `## N. Title`.
- Add subsections with `### N.x Title` only when the section is long.
- Use tables for path maps, schema fields, metrics, and stage responsibilities.
- Use fenced code blocks for commands, artifact paths, JSON examples, and SPLL examples.
- Prefer relative repo paths such as `mnist_spll_pipeline/inference_engine.py` over machine-specific absolute paths.
- Put caveats near the workflow they affect, and also summarize major ones in `Current Known Caveats / Sharp Edges`.
- When behavior changes, update the specific section and the caveat list if applicable.

### Wording rules

- Write current behavior in present tense.
- Mark uncertain or unverified claims explicitly.
- Do not claim a stage was run unless it was actually run.
- Distinguish exact behavior from thesis interpretation.
- When a field name is misleading but fixed for compatibility, document both the field name and the intended meaning.

### Patch-note rules

When a patch changes behavior, add a short note in the relevant section. Do not create a separate changelog unless the user asks; this file should stay organized by concept rather than by date.

Minimum update examples:

- New raw inference field -> update `Raw run schema highlights` and visualization metric notes.
- New figure/table -> update visualization outputs and metric descriptions.
- New config key -> update config table and caveats.
- New stage command -> update workflow commands.
- New architecture boundary -> update architecture snapshot and design guidelines.

## 12. Response Guidelines For Future Assistants

The user usually wants practical patches, not GitHub issue management.

`mnist_spll_pipeline/README.md` is intentionally a short operator guide. Keep detailed architecture notes, raw schema descriptions, caveats, and future-work discipline in this knowledge base instead of duplicating them in the README. The README must still list both Pipeline I and Pipeline II with their entrypoints, stage orders, config files, commands, and main output roots; do not delete Pipeline II while shortening the README.

Prefer this response shape for code changes:

1. short summary of what changed;
2. link to a downloadable `.patch` file;
3. exact command:

```bash
git apply <patch-file.patch>
```

4. verification performed, honestly stated.

Avoid creating GitHub issues or PRs unless the user explicitly asks. If asked for architectural planning, it is fine to propose candidates, but once the user says to act, produce a concrete patch.

## 13. Glossary

| Term | Meaning in this pipeline |
|---|---|
| Exact | SPLL compilation/inference with `cutoff = null`, no `-k` pruning flag. |
| Approximate | SPLL compilation/inference with numeric pruning threshold passed via `-k`. |
| `0.0` cutoff | Approximate code path with zero cutoff; useful for overhead comparison, not identical to exact. |
| Global cutoff | Current accumulated global path-mass pruning mode; fixed pipeline policy. |
| Adaptive top-k / posterior-mass mode | Pipeline I threshold spec or Pipeline II inference mode that mutates generated `TOP_K_CUTOFF` to target a configured surviving posterior mass, defaulting to `0.8`. |
| Runtime top-k cutoff | The selected mutable `TOP_K_CUTOFF` value used by an adaptive mode for the current model snapshot. |
| Candidate sum | One possible output sum from `0` to `9 * n_terms`. |
| Full posterior | Querying every candidate sum for one staged experiment. |
| True candidate | The ground-truth sum from labels, queried separately after the full posterior. |
| Output pool | Number of candidate sums with raw posterior mass greater than `EPS`. |
| Collapse rate | Fraction of runs where the full posterior raw mass is zero or less. |
| Branch count | Compiler-provided internal branch count when SPLL is compiled with `-c`. |
| Selected test accuracy | Existing field name for selected validation/model-selection accuracy. |

## 14. Pipeline II: training through generated SPLL inference

Pipeline II is separate from the original inference-evaluation pipeline. It tests whether exact versus approximate SPLL inference changes the **training process** itself.

### 14.1 Core idea

Each training case is one MNIST sum case. The base MNIST model is trained from the true sum only. The training loop now supports SPLL sum mini-batches controlled by `training.sum_batch_size`:

```text
for each case i in the current sum batch:
    p_true_i = generated_spll.main.forward(true_sum_i, *global_indices_i)
loss = mean_i(-log(p_true_i + epsilon))
```

The redesigned benchmark uses `sum_batch_size: 1`, so one training step is one iteration / one sum case. Pure exact and pure approximate runs train for fixed `max_steps`; there is no validation-driven early stopping.

Pipeline II keeps the old full-posterior validation machinery as optional diagnostic plumbing, but the default redesigned benchmark sets `validation.enabled: false` and does not use held-out validation for checkpointing or plots. The main measured signals are training loss and training true-sum posterior probability `p(true_sum)`.

Pipeline II rotates inference-mode run order by seed/experiment index instead of always running exact first. Run summaries record `mode_order_position`, `mode_order_offset`, and `run_order_index` so thermal/cache/load-order effects can be audited later.

The generated SPLL Python artifact is part of the gradient path. Do not replace it with a hand-written torch-native semantic equivalent for Pipeline II experiments. Batched training keeps the generated artifact as the source of truth: it precomputes differentiable MNIST softmax rows for all unique image indices in the batch, installs a temporary `readMNist(index)` lookup backed by those tensors, and still calls the generated SPLL `main.forward(...)` once per scalar sum case. The lookup tensors must remain attached to the current autograd graph.

### 14.2 Pipeline II files

| File | Role |
|---|---|
| `run_spll_training_pipeline.py` | Stage dispatcher for Pipeline II. |
| `mnist_spll_training_config.yaml` | Default Pipeline II config. |
| `mnist_spll_training_smoke_config.yaml` | Layered smoke override. |
| `prepare_spll_training.py` | Creates the balanced split, schedule manifests/previews, and shared initial checkpoints. |
| `compile_spll_training.py` | Writes SPLL source programs and compiles generated Python artifacts per arity/mode. |
| `train_spll_generated.py` | Trains through generated SPLL artifacts with differentiable `readMNist`. |
| `visualize_spll_training.py` | Writes milestone tables, per-arity grouped milestone bar charts, and trace figures. |
| `spll_training_core.py` | Shared Pipeline II helpers. |

### 14.3 Stage order and environments

Pipeline II stage order:

```text
prepare -> compile -> train -> visualize
```

Preferred Apple Silicon environment split:

| Stage | Environment |
|---|---|
| `prepare` | native arm64 torch venv |
| `compile` | Rosetta/x86 SPLL venv |
| `train` | native arm64 torch venv |
| `visualize` | native arm64 torch venv |

`all` exists for convenience but is less robust on Apple Silicon because the compile stage prefers a different architecture.

### 14.4 Data and schedules

Pipeline II uses the official MNIST train partition only. It materializes one global equal-count-per-digit 80/20 split:

- train source pool: used to sample sum-supervised training cases;
- validation pool: materialized for compatibility and optional diagnostics, but unused by the default redesigned benchmark;
- official MNIST test partition: reserved/unused.

The training schedule is compact and random-access deterministic. `prepare` writes manifests plus a small preview, not a huge JSONL schedule up to `max_steps`. A case is reproduced from `(seed, n_terms, step, split)`.

Within one sum case, repeated digits must use distinct MNIST image indices. Across steps, replacement is allowed. In the redesigned benchmark, `step` and `max_steps` are training iterations / sum cases seen.

### 14.5 Exact and approximate modes

Current Pipeline II modes include both fixed-cutoff and adaptive-posterior-mass approximation:

```yaml
adaptive_top_k:
  posterior_mass_target: 0.8
  probe_cases: 20
  max_iterations: 14
  tolerance: 0.02

inference_modes:
  - name: exact
    top_k_cutoff: null
  - name: approx_mass_0p8
    top_k_cutoff: auto
    adaptive_top_k: true
    posterior_mass_target: 0.8
  - name: approx_0p01
    top_k_cutoff: 0.01
  - name: approx_0p05
    top_k_cutoff: 0.05
  - name: approx_0p1
    top_k_cutoff: 0.1
```

`exact` means true exact compilation without `-k`. Fixed numeric modes compile and run with that numeric threshold. Adaptive modes compile through the approximate code path using `-k 0.0` as the seed artifact, then mutate the generated module-level `TOP_K_CUTOFF` at runtime. The target is the mean unnormalised posterior mass that survives pruning when the validator enumerates all candidate sums.

Adaptive cutoff search is deterministic bounded monotone bisection, not simulated annealing. This was chosen because increasing `TOP_K_CUTOFF` should only prune more branch mass, so a one-dimensional bounded search is faster, reproducible, and easier to defend experimentally. In the redesigned benchmark, the search uses deterministic train-pool probe cases from the current model snapshot, stores the selected runtime cutoff in traces/summaries, and refreshes the trainer cutoff at preflight and final bookkeeping.

Branch counting is enabled by default as sanity metadata. The loss uses only the probability component of the generated return value.

A fully pruned true-sum path may return a literal Python `0.0` from the generated artifact instead of a tensor. Pipeline II treats this as a legitimate zero-mass pruning event, converts it to a model-connected zero tensor for `-log(epsilon)` training, logs `zero_true_mass=1`, and produces a zero-gradient optimizer step. Nonzero scalar probabilities are still invalid because they would indicate a detached generated-artifact path.

### 14.6 Posterior checkpoints, traces, and fixed-budget stopping

Pipeline II trains from sum labels only. The default benchmark no longer uses task-level validation accuracy milestones. Instead, after all pure exact seeds finish, the pipeline computes aggregate exact checkpoints by averaging the exact training `p(true_sum)` trace across seeds, applying a strict full-window rolling mean over `checkpointing.rolling_window_updates` (default 50), and finding crossings of `checkpointing.posterior_thresholds`. The first 50-point rolling value is emitted only after 50 updates; shorter prefix windows are not used. These aggregate checkpoints are the source of truth for checkpoint-transfer segments.

Training stops only at the configured fixed `max_steps`. Loss and true-sum posterior are logged in `train_trace.csv` and are the primary comparison metrics. Pure exact runs also save step checkpoints under `checkpoints/steps/` so aggregate checkpoint starts can be loaded exactly for each seed. Per-run `posterior_checkpoints.json` and `milestones.json` remain as compatibility/diagnostic outputs, but checkpoint-transfer uses `aggregate_checkpoints/terms_*_exact_posterior_checkpoints.json`.

Pipeline II also supports a checkpoint-transfer recovery stage:

```bash
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_config.yaml checkpoint-transfer
```

This reruns only the green checkpoint-to-checkpoint approximate continuations. It rebuilds aggregate exact checkpoints if needed, loads the exact per-seed step checkpoint at aggregate step `s_A`, trains the approximate mode on the same deterministic training cases `s_A+1..s_B`, and stops after the same number of sum cases the aggregate exact curve needed to reach the next anchor.

Important run artifacts:

- `train_trace.csv`: one row per optimizer update. `step` is the cumulative number of sum cases seen. The row contains batch-mean loss, batch-mean true-sum mass, batch zero-mass rate, `branch_count_mean`, `branch_count_total`, batch size, optimizer update number, gradient norm, and adaptive-cutoff metadata when applicable;
- `aggregate_checkpoints/terms_*_exact_posterior_checkpoints.json`: aggregate exact rolling checkpoints and per-seed exact step checkpoint paths;
- `posterior_checkpoints.json`: per-run training true-sum posterior threshold crossings kept as diagnostic/compatibility metadata;
- `milestones.json`: backward-compatible alias for the same posterior threshold crossings;
- `checkpoint_transfer_trace.csv`: one row per approximate checkpoint-transfer segment;
- `checkpoint_transfer_train_trace.csv`: per-iteration loss/posterior trace for the green checkpoint-transfer curve;
- `checkpoints/steps/step_*.pt`: exact per-step snapshots used by aggregate checkpoint-transfer;
- `checkpoints/posterior_*.pt`: per-run posterior-threshold crossing snapshots;
- `checkpoints/final.pt`: final fixed-budget snapshot.

Visualization behavior:

- `visualize_spll_training.py` writes both `milestone_summary.*` and `run_summary.*`. The run summary makes censored, failed, and missing runs explicit, including final sum-posterior accuracy, reached-highest-milestone status, mean step time, zero-true-mass rate, and mean branch count.
- `visualize_spll_training.py` also writes `milestone_aggregate_summary.*`, which stores per-`(n_terms, mode, milestone)` reached-seed counts, means, sample standard deviations, configured uncertainty half-widths, and min/max values for steps and wall-clock time. It additionally writes `adaptive_topk_events.*` and `adaptive_topk_search_trace.*` when adaptive modes are present.
- Main-text Pipeline II comparison figures are written per `(n_terms, approximate mode)`: blue is pure exact, red is pure approximate from initialization, and green is approximate checkpoint-transfer. Loss and true-sum posterior are separate figures, not dual-axis plots. Aggregate exact posterior checkpoints are marked by distinct purple X markers on the displayed exact rolling curve; the green checkpoint-transfer curve is intentionally split into disconnected segment pieces at those boundaries.
- Pipeline II visualization uses a stable color map derived from inference-mode order. The same mode color is reused across steps-to-milestone, time-to-milestone, and trace figures.
- Main-text Pipeline II trace figures compute rolling traces per seed first, then plot the across-seed mean with configurable uncertainty bands. This avoids hiding seed variability by smoothing only after aggregation. Raw traces are still exported to appendix figures with `_raw_trace` in the file name; raw uncertainty bands are disabled by default to keep appendix plots readable. Adaptive top-k cutoff and mass traces intentionally do **not** use rolling means because the values are calibrated control settings and deterministic bisection probes; smoothing would hide refresh jumps and convergence behavior.
- Milestone figures keep unreached modes visible via summary tables and a plot footnote instead of silently pretending missing modes do not exist.
- Wall-clock milestone plots may use a log y-axis when values span more than about 5x; tick labels should remain readable scalar seconds, not opaque scientific notation.

Pipeline II visualization config:

```yaml
visualization:
  trace_smoothing_window_points: 100
  uncertainty_interval: std   # std | sem | ci95 | none
  min_uncertainty_samples: 2
  show_milestone_error_bars: true
  show_trace_uncertainty_bands: true
  show_raw_trace_uncertainty_bands: false
  trace_band_alpha: 0.16
```

The default uncertainty interval is `std`, interpreted as one sample standard deviation across seeds. `sem` and `ci95` are available for narrower uncertainty-of-the-mean displays, but with the default five seeds `std` is the more conservative thesis-facing default.

### 14.7 Safety rails

Pipeline II should fail loudly rather than silently produce invalid training results. The train stage should abort if:

- generated artifacts are missing;
- the generated probability is not a differentiable torch tensor, except for literal zero from a fully pruned true-sum path;
- the preflight gradient check cannot find at least one nonzero-gradient case across its probe cases;
- probability/loss becomes NaN, infinite, or negative for any case in a sum batch;
- gradients contain NaN or infinity.

The train stage must not auto-compile missing artifacts. It should tell the user to run the compile stage in the Rosetta/x86 environment.

### 14.8 Config inheritance

`mnist_spll_common.load_config(...)` supports exactly one `extends` level. Dicts deep-merge and lists replace completely. Nested inheritance is rejected.

This is used for the Pipeline II smoke config. Whenever the default Pipeline II config changes, the smoke override must be checked and updated in the same patch so it remains a valid fast representative of the default experiment.

## 15. Patch discipline

Every patch/change that affects pipeline behavior, file structure, workflow, output schemas, config semantics, or known caveats must update this knowledge base in the same patch.

For Pipeline II specifically, README updates are mandatory whenever user-facing commands, stages, config names, environment assumptions, or expected outputs change.
